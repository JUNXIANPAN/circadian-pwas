"""
engine_api.py
─────────────
薄包装层：把 simulation3_4 的 CLI 流水线封装成一个 Python 函数，
供 Streamlit 前端和 Agent 直接调用，无需 subprocess。

用法
----
from engine_api import run_pwas_simulation

result = run_pwas_simulation({
    "n_reps":   20,
    "tau":      1.0,
    "h2_med":   0.20,
    "h2_non_med": 0.10,
    "n_causal": 20,
    "use_real_ld": False,
})

返回 dict，见 Returns 说明。
"""

import os
import sys
import numpy as np
import pandas as pd

# ── 把 simulation3_4 所在目录加入 sys.path ──────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import simulation3_4 as _sim

# 默认数据路径（和 simulation3_4 保持一致）
_PWAS_ROOT  = os.path.abspath(os.path.join(_HERE, "..", ".."))
_D_PROFILES = os.path.join(_PWAS_ROOT, "raw_data", "circadian_info", "report.pg_matrix.tsv")
_D_PQTL     = os.path.join(_PWAS_ROOT, "work", "pqtl_topk.csv")
_D_LDREF    = os.path.join(_PWAS_ROOT, "raw_data", "pQTL", "ldref", "LDREF")
_D_UKBB_LD  = "/data/CommonData/ukbb-ld"
_D_CHAIN    = os.path.join(_PWAS_ROOT, "reference", "chain", "hg38ToHg19.over.chain.gz")


def run_pwas_simulation(params: dict, cache_dir: str | None = None) -> dict:
    """
    运行节律加权 PWAS 模拟，返回结构化结果。

    Parameters
    ----------
    params : dict
        可选键（未提供则用默认值）：
          n_reps        int    重复次数（默认 20，快速测试用）
          tau           float  节律权重锐度（默认 1.0）
          h2_med        float  蛋白介导遗传率（默认 0.20）
          h2_non_med    float  直接遗传率（默认 0.10）
          n_causal      int    因果蛋白数（默认 20）
          n_direct_snps int    直接效应 SNP 数（默认 50）
          use_real_ld   bool   使用 UKBB-LD（默认 False，快）
          use_metacycle bool   使用 MetaCycle（默认 False，用 cosinor 代替）
          profiles_path str    蛋白节律文件路径
          pqtl_path     str    pQTL 文件路径
          seed          int    随机种子（默认 42）

    cache_dir : str | None
        缓存目录（liftover、MAF 等）。None 则用临时目录。

    Returns
    -------
    dict with keys:
      "scenario_summary"  : pd.DataFrame  三场景 ΔAuC（mean ± SE）
      "circadian_delta_auc" : float       circadian 场景平均 ΔAuC
      "null_delta_auc"      : float       null 场景平均 ΔAuC
      "random_delta_auc"    : float       random 场景平均 ΔAuC
      "tagging_summary"   : dict | None   tagging 假阳性率（若有）
      "top_proteins"      : list[str]     加权排名前10的蛋白名
      "corr_r2_snr"       : float         R² 与真实 SNR 的相关（诊断用）
      "n_reps"            : int
      "tau"               : float
      "raw"               : dict          run_simulation() 完整返回值
    """
    # ── 构建 Config ────────────────────────────────────────────────────────────
    cfg = _sim.Config()
    cfg.n_reps        = int(params.get("n_reps",        20))
    cfg.tau           = float(params.get("tau",          1.0))
    cfg.h2_med        = float(params.get("h2_med",       0.20))
    cfg.h2_non_med    = float(params.get("h2_non_med",   0.10))
    cfg.n_causal      = int(params.get("n_causal",       20))
    cfg.n_direct_snps = int(params.get("n_direct_snps",  50))
    cfg.use_real_ld   = bool(params.get("use_real_ld",   False))
    cfg.use_metacycle = bool(params.get("use_metacycle", False))
    cfg.seed          = int(params.get("seed",           42))

    profiles_path = params.get("profiles_path", _D_PROFILES)
    pqtl_path     = params.get("pqtl_path",     _D_PQTL)

    # ── 数据文件存在性检查（云端部署时优雅降级）──────────────────────────────────
    if not os.path.exists(profiles_path) or not os.path.exists(pqtl_path):
        return {
            "scenario_summary":    None,
            "circadian_delta_auc": float("nan"),
            "null_delta_auc":      float("nan"),
            "random_delta_auc":    float("nan"),
            "tagging_summary":     None,
            "top_proteins":        [],
            "corr_r2_snr":         float("nan"),
            "n_reps":              0,
            "tau":                 params.get("tau", 1.0),
            "raw":                 {},
            "_unavailable":        True,
            "_reason": (
                "模拟引擎需要服务器上的 pQTL 和蛋白节律数据文件，"
                "云端部署模式下无法运行完整模拟。"
                "文献检索和蛋白注释功能照常可用。"
            ),
        }

    if cache_dir is None:
        import tempfile
        cache_dir = tempfile.mkdtemp(prefix="pwas_cache_")

    os.makedirs(cache_dir, exist_ok=True)

    # ── Step 1: 学节律参数分布 ─────────────────────────────────────────────────
    rp = _sim.learn_rhythm_params(profiles_path)

    # ── Step 2: 加载 pQTL ─────────────────────────────────────────────────────
    pqtl_df = _sim.load_pqtl(pqtl_path)

    # ── Step 3: MAF（快速模式用随机 MAF，不走 liftover / UKBB-LD）────────────
    if cfg.use_real_ld:
        maf_series, _ = _sim.load_maf_ld(
            pqtl_df, _D_LDREF,
            chain_file=_D_CHAIN,
            ukbb_dir=_D_UKBB_LD,
            cache_dir=cache_dir,
        )
    else:
        snps = pqtl_df["snp_id"].unique()
        # 尝试读已有的 ensembl MAF 缓存
        ensembl_cache = os.path.join(
            _PWAS_ROOT, "raw_data", "reference", "ensembl_maf.csv")
        if os.path.exists(ensembl_cache):
            emaf = pd.read_csv(ensembl_cache).set_index("rsid")["maf"]
            maf_arr = np.array([
                float(emaf.get(s, np.random.default_rng(42).uniform(0.05, 0.45)))
                for s in snps
            ])
        else:
            maf_arr = np.random.default_rng(cfg.seed).uniform(0.05, 0.45, len(snps))
        maf_series = pd.Series(maf_arr, index=snps)

    # ── Step 4: 整合 ──────────────────────────────────────────────────────────
    data = _sim.harmonize(pqtl_df, maf_series)
    data["LD"] = None   # 快速模式跳过 LD

    if cfg.use_real_ld and os.path.isdir(_D_UKBB_LD):
        _sim.load_ukbb_ld(data, _D_UKBB_LD, cache_dir=cache_dir)

    # ── Step 5: 运行模拟 ──────────────────────────────────────────────────────
    res = _sim.run_simulation(cfg, data, rp)

    # ── 整理返回值 ────────────────────────────────────────────────────────────
    summ = res["summary"].set_index("scenario")

    def _safe(sc, col="mean_delta_auc"):
        try:
            return float(summ.loc[sc, col])
        except Exception:
            return float("nan")

    # top proteins：从 snapshot 取加权排名前10
    top_proteins = []
    if res.get("snapshot") is not None:
        snap = res["snapshot"]
        top_proteins = snap.head(10)["protein"].tolist()

    # R² 与 SNR 的相关（诊断指标）
    corr_r2_snr = float("nan")
    if res.get("snr") is not None and res.get("r2") is not None:
        snr_arr = res["snr"]
        r2_arr  = res["r2"]
        if len(snr_arr) == len(r2_arr) and len(snr_arr) > 1:
            corr_r2_snr = float(np.corrcoef(snr_arr, r2_arr)[0, 1])

    return {
        "scenario_summary":    res["summary"],
        "circadian_delta_auc": _safe("circadian"),
        "null_delta_auc":      _safe("null"),
        "random_delta_auc":    _safe("random"),
        "tagging_summary":     res.get("tagging_summary"),
        "top_proteins":        top_proteins,
        "corr_r2_snr":         corr_r2_snr,
        "n_reps":              cfg.n_reps,
        "tau":                 cfg.tau,
        "raw":                 res,
    }
