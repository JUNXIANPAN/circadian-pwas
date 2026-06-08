"""
orchestrator.py
───────────────
总控 Agent：把用户的自然语言问题解析成 tool calls，
依次调度文献 Agent、仿真 Agent、注释 Agent、报告 Agent。

真实模式：需要 ANTHROPIC_API_KEY 环境变量。
Mock 模式：无需 API key，返回固定示例数据（用于开发调试）。
"""

import os
import json
import re
from typing import Optional

_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
_USE_MOCK = not bool(_API_KEY)

if not _USE_MOCK:
    import anthropic
    _client = anthropic.Anthropic(api_key=_API_KEY)


# ── Tool 定义（供总控 Agent 使用）────────────────────────────────────────────
_TOOLS = [
    {
        "name": "check_circadian_trait",
        "description": (
            "检索文献，判断给定性状是否具有昼夜节律性，并返回验证过的 PMID 引用。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "trait_name": {
                    "type": "string",
                    "description": "要检查的性状名称，英文或中文均可"
                }
            },
            "required": ["trait_name"]
        }
    },
    {
        "name": "run_simulation",
        "description": "调用节律加权 PWAS 仿真引擎，返回三场景 ΔAuC 等结果。",
        "input_schema": {
            "type": "object",
            "properties": {
                "n_reps":    {"type": "integer", "description": "重复次数，默认 20"},
                "tau":       {"type": "number",  "description": "节律权重强度，默认 1.0"},
                "h2_med":    {"type": "number",  "description": "蛋白介导遗传率，默认 0.20"},
                "n_causal":  {"type": "integer", "description": "因果蛋白数，默认 20"},
            },
            "required": []
        }
    },
    {
        "name": "annotate_proteins",
        "description": "查询 Open Targets 和 GWAS Catalog，注释给定蛋白的已知疾病关联和可成药性。",
        "input_schema": {
            "type": "object",
            "properties": {
                "proteins": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "蛋白基因名列表，如 ['APOB', 'PCSK9']"
                }
            },
            "required": ["proteins"]
        }
    },
    {
        "name": "generate_report",
        "description": "根据前几步的结果生成大白话分析报告，标记置信度低的部分。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":       {"type": "string", "description": "用户原始问题"},
                "lit_result":  {"type": "object", "description": "文献检索结果"},
                "sim_result":  {"type": "object", "description": "模拟结果"},
                "ann_result":  {"type": "array",  "description": "蛋白注释结果"},
            },
            "required": ["query"]
        }
    },
]


# ── 工具执行函数 ──────────────────────────────────────────────────────────────
def check_circadian_trait(query: str) -> dict:
    """判断性状节律性，返回 {is_circadian, confidence, evidence, trait_name}"""
    if _USE_MOCK:
        return _mock_literature(query)
    from agents.literature_agent import check_circadian_trait as _lit
    return _lit(query)


def annotate_proteins(proteins: list[str]) -> list[dict]:
    """查 Open Targets + GWAS Catalog 注释蛋白，返回注释列表"""
    from agents.annotation_agent import annotate_proteins as _ann
    return _ann(proteins)


def generate_report(query: str, lit_result, sim_result, ann_result) -> str:
    """用 LLM 生成大白话报告"""
    if _USE_MOCK:
        return _mock_report(query, sim_result)
    return _llm_report(query, lit_result, sim_result, ann_result)


# ── 真实 LLM 报告 ─────────────────────────────────────────────────────────────
def _llm_report(query: str, lit, sim, ann) -> str:
    c_auc = sim["circadian_delta_auc"] if sim else float("nan")

    context = f"""
用户问题：{query}

文献检索：{json.dumps(lit, ensure_ascii=False, default=str) if lit else '未运行'}

模拟结果：
- circadian ΔAuC = {c_auc:+.4f}
- null      ΔAuC = {sim['null_delta_auc']:+.4f if sim else 'N/A'}
- random    ΔAuC = {sim['random_delta_auc']:+.4f if sim else 'N/A'}
- top 蛋白  = {sim['top_proteins'][:5] if sim else []}

蛋白注释：{json.dumps(ann[:3] if ann else [], ensure_ascii=False, default=str)}
"""

    resp = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                "你是一位生物统计学专家助手，请根据以下分析结果写一份简明的中文报告（300字以内）。"
                "对置信度低或证据不足的地方，用⚠️标记。\n\n" + context
            )
        }]
    )
    return resp.content[0].text


# ── Mock 实现（无需 API key）──────────────────────────────────────────────────
def _mock_literature(query: str) -> dict:
    keywords = ["睡眠", "sleep", "昼夜", "circadian", "节律", "血压", "pressure",
                "皮质醇", "cortisol", "褪黑素", "melatonin", "chronotype",
                "triglyceride", "甘油三酯", "glucose", "diurnal", "rhythm"]
    is_circ = any(k in query.lower() for k in keywords)
    return {
        "is_circadian": is_circ,
        "confidence": 0.82 if is_circ else 0.35,
        "trait_name": query[:30],
        "evidence": [
            {"pmid": "25135935", "title": "Circadian rhythms in blood pressure regulation",
             "year": 2014},
            {"pmid": "31589406", "title": "The circadian clock and human health",
             "year": 2019},
        ] if is_circ else [],
        "_mock": True,
    }


def _mock_annotation(proteins: list[str]) -> list[dict]:
    known = {
        "APOB":  ("Cardiovascular disease", True),
        "PCSK9": ("Hypercholesterolemia", True),
        "CRP":   ("Inflammation", False),
        "APOA1": ("Cardiovascular disease", True),
    }
    rows = []
    for p in proteins[:10]:
        disease, druggable = known.get(p, ("Unknown", False))
        rows.append({
            "蛋白": p,
            "已知疾病关联": disease,
            "可成药": "✅" if druggable else "—",
            "数据来源": "Open Targets (mock)",
        })
    return rows


def _mock_report(query: str, sim) -> str:
    if sim is None:
        return "⚠️ 模拟未完成，无法生成报告。"
    c = sim["circadian_delta_auc"]
    direction = "有所提升" if c > 0.01 else ("基本持平" if c > -0.01 else "略有下降")
    flag = "" if abs(c) > 0.005 else "⚠️ 效果不显著，建议增加重复次数或检查性状节律证据。"
    return f"""
**问题**：{query}

**结论**：节律加权 PWAS 在 circadian 场景下 ΔAuC = **{c:+.4f}**，相比传统 PWAS **{direction}**。{flag}

| 场景 | ΔAuC | 含义 |
|------|------|------|
| Circadian | {sim['circadian_delta_auc']:+.4f} | 方法在节律性状上的表现 |
| Null | {sim['null_delta_auc']:+.4f} | 无信号时不产生假提升 ✓ |
| Random | {sim['random_delta_auc']:+.4f} | 用错对象时有代价（正常） |

**Top 命中蛋白**：{', '.join(sim['top_proteins'][:5]) if sim['top_proteins'] else '无'}

> *本报告由 Mock Agent 生成（未使用 Anthropic API）。设置 ANTHROPIC_API_KEY 后可获得真实 AI 报告。*
"""
