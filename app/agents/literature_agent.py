"""
literature_agent.py
───────────────────
判断给定性状是否具有昼夜节律性，返回带真实 PMID 的支持证据。

防幻觉策略
----------
1. PMID 全部来自 PubMed E-utilities esearch，不由 LLM 生成
2. 每个 PMID 必须经 esummary 回查确认标题和年份存在
3. Claude 只负责：①提取性状英文名 ②根据真实摘要判断节律性

依赖
----
- anthropic（Claude API）：需要 ANTHROPIC_API_KEY
- requests：标准库外唯一依赖
- NCBI_API_KEY（可选）：设置后 PubMed 请求限速从 3/s 升至 10/s
"""

import os
import time
import requests
import json
from typing import Optional

_ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
_NCBI_KEY      = os.environ.get("NCBI_API_KEY", "")

_ESEARCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
_EFETCH_URL   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


# ── 主入口 ────────────────────────────────────────────────────────────────────
def check_circadian_trait(query: str, max_papers: int = 8) -> dict:
    """
    判断查询中提到的性状是否具有昼夜节律性。

    Parameters
    ----------
    query      : 用户自然语言问题（中文或英文）
    max_papers : 从 PubMed 拉取的最大论文数

    Returns
    -------
    {
      is_circadian : bool,
      confidence   : float (0-1),
      trait_name   : str,          # 提取出的英文性状名
      evidence     : list[dict],   # 已验证论文 [{pmid, title, year, abstract_snippet}]
      reasoning    : str,          # Claude 的判断理由
      _source      : str           # "real" or "mock"
    }
    """
    if not _ANTHROPIC_KEY:
        raise RuntimeError(
            "未设置 ANTHROPIC_API_KEY，无法使用真实文献 Agent。"
            "请 export ANTHROPIC_API_KEY=... 或使用 Mock 模式。"
        )

    import anthropic
    client = anthropic.Anthropic(api_key=_ANTHROPIC_KEY)

    # Step 1: 用 Claude 提取性状英文名
    trait_en = _extract_trait_name(client, query)

    # Step 2: PubMed esearch 拿 PMID
    pmids = _pubmed_search(trait_en, max_results=max_papers)
    if not pmids:
        return {
            "is_circadian": False,
            "confidence": 0.1,
            "trait_name": trait_en,
            "evidence": [],
            "reasoning": f"PubMed 中未找到 '{trait_en} circadian' 相关文献。",
            "_source": "real",
        }

    # Step 3: esummary 拿标题 + 年份（验证 PMID 真实存在）
    papers = _pubmed_summaries(pmids)

    # Step 4: efetch 拿摘要片段（用于给 Claude 看）
    papers = _enrich_abstracts(papers, max_chars=300)

    # Step 5: Claude 根据真实摘要判断节律性
    result = _assess_circadian(client, trait_en, papers)
    result["trait_name"] = trait_en
    result["_source"]    = "real"
    return result


# ── Step 1: 提取性状英文名 ────────────────────────────────────────────────────
def _extract_trait_name(client, query: str) -> str:
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=64,
        messages=[{
            "role": "user",
            "content": (
                "从下面的问题中提取用户想研究的性状名称，用标准英文医学术语回答，"
                "只回答性状名，不要其他内容。\n\n问题：" + query
            )
        }]
    )
    trait = resp.content[0].text.strip().strip('"').strip("'")
    # 截短防止查询太长
    return trait[:60] if len(trait) > 60 else trait


# ── Step 2: PubMed esearch ────────────────────────────────────────────────────
def _pubmed_search(trait_en: str, max_results: int = 8) -> list[str]:
    """搜索 '{trait} circadian rhythm'，返回 PMID 列表。"""
    search_term = f"{trait_en} circadian rhythm"
    params = {
        "db":      "pubmed",
        "term":    search_term,
        "retmax":  max_results,
        "retmode": "json",
        "sort":    "relevance",
    }
    if _NCBI_KEY:
        params["api_key"] = _NCBI_KEY

    try:
        r = requests.get(_ESEARCH_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        return pmids
    except Exception as e:
        print(f"[literature_agent] PubMed esearch 失败: {e}")
        return []


# ── Step 3: PubMed esummary（验证 PMID + 拿元数据）─────────────────────────
def _pubmed_summaries(pmids: list[str]) -> list[dict]:
    """根据 PMID 列表获取标题和年份，验证 PMID 真实存在。"""
    if not pmids:
        return []

    params = {
        "db":      "pubmed",
        "id":      ",".join(pmids),
        "retmode": "json",
    }
    if _NCBI_KEY:
        params["api_key"] = _NCBI_KEY

    try:
        r = requests.get(_ESUMMARY_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        result = data.get("result", {})
        papers = []
        for pmid in pmids:
            info = result.get(pmid, {})
            if not info or info.get("error"):
                continue  # PMID 不存在或错误，跳过
            title = info.get("title", "").rstrip(".")
            year  = info.get("pubdate", "")[:4]
            if title:  # 只保留有标题的
                papers.append({
                    "pmid":    pmid,
                    "title":   title,
                    "year":    year,
                    "abstract_snippet": "",  # 后续 efetch 补充
                })
        return papers
    except Exception as e:
        print(f"[literature_agent] PubMed esummary 失败: {e}")
        return []


# ── Step 4: efetch 摘要（可选，让 Claude 判断更准）────────────────────────────
def _enrich_abstracts(papers: list[dict], max_chars: int = 300) -> list[dict]:
    """为每篇论文补充摘要片段（前 max_chars 字符）。"""
    if not papers:
        return papers

    pmids = [p["pmid"] for p in papers]
    params = {
        "db":       "pubmed",
        "id":       ",".join(pmids),
        "rettype":  "abstract",
        "retmode":  "text",
    }
    if _NCBI_KEY:
        params["api_key"] = _NCBI_KEY

    try:
        r = requests.get(_EFETCH_URL, params=params, timeout=15)
        r.raise_for_status()
        text = r.text

        # 按 PMID 分割，简单提取每段的前几行作为摘要片段
        blocks = text.split("\n\n")
        # 把摘要文本分配给 papers（顺序对应）
        for i, paper in enumerate(papers):
            snippet = ""
            for block in blocks:
                if paper["pmid"] in block[:50]:
                    # 找到对应块，取之后的文字
                    idx = blocks.index(block)
                    for j in range(idx + 1, min(idx + 4, len(blocks))):
                        snippet += blocks[j] + " "
                        if len(snippet) > max_chars:
                            break
                    break
            paper["abstract_snippet"] = snippet[:max_chars].strip()
    except Exception as e:
        print(f"[literature_agent] efetch 失败（非致命）: {e}")

    return papers


# ── Step 5: Claude 判断节律性 ─────────────────────────────────────────────────
def _assess_circadian(client, trait_en: str, papers: list[dict]) -> dict:
    """让 Claude 根据真实论文标题/摘要判断性状是否具有节律性。"""
    if not papers:
        return {
            "is_circadian": False,
            "confidence": 0.1,
            "evidence": [],
            "reasoning": "未找到相关文献，无法判断。",
        }

    # 构建论文摘要供 Claude 阅读
    paper_text = "\n".join([
        f"- PMID {p['pmid']} ({p['year']}): {p['title']}"
        + (f"\n  摘要片段: {p['abstract_snippet']}" if p['abstract_snippet'] else "")
        for p in papers
    ])

    prompt = f"""你是一位生物节律领域专家。

性状名称：{trait_en}

以下是从 PubMed 搜索 "{trait_en} circadian rhythm" 得到的真实论文（PMID 已验证）：

{paper_text}

请根据以上文献判断：
1. 该性状是否具有明显的昼夜节律性（circadian rhythm）？
2. 置信度是多少（0-1）？
3. 简要说明理由（1-2句话）。

请用以下 JSON 格式回答，不要输出其他内容：
{{
  "is_circadian": true 或 false,
  "confidence": 0.0-1.0,
  "reasoning": "理由"
}}"""

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )

    text = resp.content[0].text.strip()
    # 提取 JSON（防止 Claude 在 JSON 前后加了多余文字）
    try:
        # 找第一个 { 到最后一个 }
        start = text.index("{")
        end   = text.rindex("}") + 1
        parsed = json.loads(text[start:end])
    except Exception:
        # 解析失败时用保守默认值
        parsed = {
            "is_circadian": False,
            "confidence":   0.3,
            "reasoning":    text[:200],
        }

    return {
        "is_circadian": bool(parsed.get("is_circadian", False)),
        "confidence":   float(max(0.0, min(1.0, parsed.get("confidence", 0.3)))),
        "evidence":     papers,
        "reasoning":    parsed.get("reasoning", ""),
    }
