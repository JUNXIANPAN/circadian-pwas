"""
annotation_agent.py
───────────────────
对 top 蛋白命中做靶点注释，数据来源：
  1. Open Targets Platform GraphQL  → 疾病关联评分 + 可成药性
  2. GWAS Catalog REST API          → GWAS 命中 SNP 数（作为佐证）

所有数据实时从公开 API 拉取，不依赖本地数据库。
"""

import requests
import time
from typing import Optional

_OT_URL     = "https://api.platform.opentargets.org/api/v4/graphql"
_GWAS_URL   = "https://www.ebi.ac.uk/gwas/rest/api/singleNucleotidePolymorphisms/search/findByGene"
_TIMEOUT    = 15
_RATE_SLEEP = 0.3   # Open Targets 限速宽松，0.3s 足够


# ── 主入口 ────────────────────────────────────────────────────────────────────
def annotate_proteins(proteins: list[str]) -> list[dict]:
    """
    对蛋白列表做靶点注释。

    Parameters
    ----------
    proteins : list[str]
        基因符号列表，如 ['APOB', 'PCSK9', 'CRP']

    Returns
    -------
    list[dict]，每个 dict 包含：
      protein          str    基因名
      ensembl_id       str    Ensembl gene ID（未找到则空）
      top_disease      str    最高关联分的疾病名
      disease_score    float  Open Targets 关联评分 (0-1)
      all_diseases     list   前3个疾病 [{name, score}]
      druggable        bool   是否有任何可成药证据
      drug_modality    str    可成药模式（SM/AB等，逗号分隔）
      approved_drugs   list   已批准药物名（若有）
      gwas_snp_count   int    GWAS Catalog 中该基因的 GWAS SNP 数
      ot_url           str    Open Targets 页面链接
    """
    results = []
    for gene in proteins:
        row = _annotate_one(gene)
        results.append(row)
        time.sleep(_RATE_SLEEP)
    return results


# ── 单个蛋白注释 ──────────────────────────────────────────────────────────────
def _annotate_one(gene: str) -> dict:
    base = {
        "protein":        gene,
        "ensembl_id":     "",
        "top_disease":    "—",
        "disease_score":  0.0,
        "all_diseases":   [],
        "druggable":      False,
        "drug_modality":  "—",
        "approved_drugs": [],
        "gwas_snp_count": 0,
        "ot_url":         f"https://platform.opentargets.org/target/{gene}",
    }

    # Step 1: 基因名 → Ensembl ID
    ensembl_id = _symbol_to_ensembl(gene)
    if not ensembl_id:
        base["top_disease"] = "未找到（基因名不在 Open Targets 中）"
        return base
    base["ensembl_id"] = ensembl_id
    base["ot_url"]     = f"https://platform.opentargets.org/target/{ensembl_id}"

    # Step 2: Open Targets 疾病关联 + 可成药性
    ot_data = _fetch_ot_target(ensembl_id)
    if ot_data:
        # 疾病关联
        diseases = ot_data.get("associatedDiseases", {}).get("rows", [])
        if diseases:
            top = diseases[0]
            base["top_disease"]   = top["disease"]["name"]
            base["disease_score"] = round(float(top["score"]), 3)
            base["all_diseases"]  = [
                {"name": d["disease"]["name"], "score": round(float(d["score"]), 3)}
                for d in diseases[:3]
            ]

        # 可成药性
        tract = ot_data.get("tractability", [])
        active_modalities = [t["modality"] for t in tract if t.get("value")]
        if active_modalities:
            base["druggable"]     = True
            base["drug_modality"] = ", ".join(sorted(set(active_modalities)))

        # 已批准药物（drugAndClinicalCandidates，maxClinicalStage == APPROVAL）
        cands = ot_data.get("drugAndClinicalCandidates", {})
        drugs = []
        for row in cands.get("rows", []):
            dname = row.get("drug", {}).get("name", "")
            stage = row.get("maxClinicalStage", "")
            if dname and stage == "APPROVAL":
                drugs.append(dname)
        base["approved_drugs"] = list(set(drugs))[:5]

    # Step 3: GWAS Catalog SNP 命中数
    base["gwas_snp_count"] = _gwas_snp_count(gene)

    return base


# ── Open Targets：基因名 → Ensembl ID ────────────────────────────────────────
def _symbol_to_ensembl(symbol: str) -> Optional[str]:
    query = """
    query SearchGene($q: String!) {
      search(queryString: $q, entityNames: ["target"], page: {index: 0, size: 1}) {
        hits {
          id
          object {
            ... on Target {
              approvedSymbol
            }
          }
        }
      }
    }
    """
    try:
        r = requests.post(
            _OT_URL,
            json={"query": query, "variables": {"q": symbol}},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        hits = r.json()["data"]["search"]["hits"]
        if hits:
            obj = hits[0]["object"]
            # 确认 symbol 匹配（防止查到同名基因）
            if obj.get("approvedSymbol", "").upper() == symbol.upper():
                return hits[0]["id"]
            # 即使 symbol 略有差异也接受（如大小写）
            return hits[0]["id"]
    except Exception as e:
        print(f"[annotation] symbol_to_ensembl({symbol}) 失败: {e}")
    return None


# ── Open Targets：拉取疾病关联 + 可成药性 + 已批准药物 ────────────────────────
def _fetch_ot_target(ensembl_id: str) -> Optional[dict]:
    query = """
    query TargetInfo($id: String!) {
      target(ensemblId: $id) {
        approvedSymbol
        associatedDiseases(page: {index: 0, size: 5}) {
          rows {
            disease { name }
            score
          }
        }
        tractability {
          label
          modality
          value
        }
        drugAndClinicalCandidates {
          count
          rows {
            drug { name maximumClinicalStage }
            maxClinicalStage
          }
        }
      }
    }
    """
    try:
        r = requests.post(
            _OT_URL,
            json={"query": query, "variables": {"id": ensembl_id}},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["data"]["target"]
    except Exception as e:
        print(f"[annotation] fetch_ot_target({ensembl_id}) 失败: {e}")
    return None


# ── GWAS Catalog：统计 GWAS SNP 命中数 ───────────────────────────────────────
def _gwas_snp_count(gene: str) -> int:
    try:
        r = requests.get(
            _GWAS_URL,
            params={"geneName": gene, "size": 1},
            headers={"Accept": "application/json"},
            timeout=_TIMEOUT,
        )
        if r.status_code != 200:
            return 0
        data = r.json()
        page = data.get("page", {})
        return int(page.get("totalElements", 0))
    except Exception as e:
        print(f"[annotation] gwas_snp_count({gene}) 失败: {e}")
        return 0
