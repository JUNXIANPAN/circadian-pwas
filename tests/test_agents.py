"""
tests/test_agents.py
────────────────────
单元测试：不依赖 API key，全部 mock 模式。
"""

import sys
import os
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "app"))
sys.path.insert(0, os.path.join(_ROOT, "scripts", "simulation3"))


# ── 文献 Agent mock 测试 ──────────────────────────────────────────────────────
class TestLiteratureMock:
    def setup_method(self):
        # 确保没有 API key，走 mock
        os.environ.pop("ANTHROPIC_API_KEY", None)
        from agents.orchestrator import check_circadian_trait
        self.check = check_circadian_trait

    def test_sleep_is_circadian(self):
        r = self.check("睡眠时长")
        assert r["is_circadian"] is True
        assert r["confidence"] > 0.5

    def test_eye_color_not_circadian(self):
        r = self.check("eye color genetics")
        assert r["is_circadian"] is False
        assert r["confidence"] < 0.6

    def test_returns_required_keys(self):
        r = self.check("cortisol levels")
        for key in ["is_circadian", "confidence", "evidence", "trait_name"]:
            assert key in r, f"missing key: {key}"

    def test_confidence_in_range(self):
        r = self.check("blood pressure")
        assert 0.0 <= r["confidence"] <= 1.0


# ── PubMed API 测试（不需要 API key）─────────────────────────────────────────
class TestPubMed:
    def setup_method(self):
        from agents.literature_agent import _pubmed_search, _pubmed_summaries
        self.search    = _pubmed_search
        self.summaries = _pubmed_summaries

    def test_sleep_search_returns_pmids(self):
        pmids = self.search("sleep duration", max_results=3)
        assert isinstance(pmids, list)
        assert len(pmids) > 0
        assert all(p.isdigit() for p in pmids)

    def test_pmid_summaries_have_titles(self):
        pmids = self.search("circadian rhythm", max_results=2)
        papers = self.summaries(pmids)
        assert len(papers) > 0
        for p in papers:
            assert "pmid" in p
            assert "title" in p
            assert len(p["title"]) > 5

    def test_nonsense_query_returns_list(self):
        # 即使查不到也应返回列表不报错
        pmids = self.search("xyzxyznonexistentterm12345", max_results=3)
        assert isinstance(pmids, list)


# ── 注释 Agent 测试 ───────────────────────────────────────────────────────────
class TestAnnotation:
    def setup_method(self):
        from agents.annotation_agent import _symbol_to_ensembl, _gwas_snp_count
        self.to_ensembl   = _symbol_to_ensembl
        self.gwas_count   = _gwas_snp_count

    def test_apob_ensembl_id(self):
        eid = self.to_ensembl("APOB")
        assert eid is not None
        assert eid.startswith("ENSG")

    def test_unknown_gene_returns_none(self):
        eid = self.to_ensembl("NOTAREALGENE99999")
        # 可能返回 None 或某个意外匹配，但不应报错
        assert eid is None or eid.startswith("ENSG")

    def test_gwas_snp_count_positive(self):
        n = self.gwas_count("APOB")
        assert isinstance(n, int)
        assert n > 0

    def test_gwas_snp_count_unknown(self):
        n = self.gwas_count("NOTAREALGENE99999")
        assert isinstance(n, int)
        assert n == 0


# ── 引擎 API 测试 ─────────────────────────────────────────────────────────────
class TestEngineAPI:
    def setup_method(self):
        from engine_api import run_pwas_simulation
        self.run = run_pwas_simulation

    def test_returns_required_keys(self):
        res = self.run({"n_reps": 3, "use_real_ld": False, "use_metacycle": False})
        for key in ["circadian_delta_auc", "null_delta_auc", "random_delta_auc",
                    "top_proteins", "scenario_summary"]:
            assert key in res, f"missing key: {key}"

    def test_delta_auc_are_floats(self):
        res = self.run({"n_reps": 3, "use_real_ld": False, "use_metacycle": False})
        assert isinstance(res["circadian_delta_auc"], float)
        assert isinstance(res["null_delta_auc"], float)
        assert isinstance(res["random_delta_auc"], float)

    def test_null_delta_auc_near_zero(self):
        res = self.run({"n_reps": 10, "use_real_ld": False, "use_metacycle": False})
        assert abs(res["null_delta_auc"]) < 0.20, \
            f"null ΔAuC = {res['null_delta_auc']:.4f}，应接近 0"

    def test_top_proteins_nonempty(self):
        res = self.run({"n_reps": 3, "use_real_ld": False, "use_metacycle": False})
        assert len(res["top_proteins"]) > 0

    def test_custom_tau(self):
        res = self.run({"n_reps": 3, "tau": 2.0,
                        "use_real_ld": False, "use_metacycle": False})
        assert res["n_reps"] == 3
        assert res["tau"] == 2.0
