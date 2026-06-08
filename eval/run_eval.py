"""
eval/run_eval.py
────────────────
评测节律加权 PWAS Agent 流水线的三个维度：

1. 性状节律性分类准确率（基于 mock 关键词规则）
2. 蛋白注释 API 可达性（Open Targets + GWAS Catalog）
3. 引擎基本健全性（n_reps=5 快速跑通）

用法
----
cd /data2/pan/pwas
python eval/run_eval.py [--full]   # --full 启用真实 API（需 ANTHROPIC_API_KEY）
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "app"))
sys.path.insert(0, os.path.join(_ROOT, "scripts", "simulation3"))

CASES_PATH  = os.path.join(os.path.dirname(__file__), "test_cases.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")


# ── 评测1：性状节律性分类 ─────────────────────────────────────────────────────
def eval_classification(full: bool = False) -> dict:
    from agents.orchestrator import check_circadian_trait

    with open(CASES_PATH) as f:
        cases = json.load(f)

    results = []
    t0 = time.time()

    for case in cases:
        t_start = time.time()
        try:
            res = check_circadian_trait(case["query"])
            predicted  = res["is_circadian"]
            confidence = res.get("confidence", 0.0)
            source     = res.get("_source", res.get("_mock") and "mock" or "real")
        except Exception as e:
            predicted  = None
            confidence = 0.0
            source     = f"error: {e}"

        expected = case["expected"]
        # ambiguous cases (expected=null) 不计入准确率
        correct = None if expected is None else (predicted == expected)

        results.append({
            "query":      case["query"],
            "category":   case["category"],
            "expected":   expected,
            "predicted":  predicted,
            "confidence": round(confidence, 3),
            "correct":    correct,
            "latency_s":  round(time.time() - t_start, 2),
            "source":     source,
        })
        print(f"  {'✓' if correct else ('?' if correct is None else '✗')} "
              f"[{case['category']:9s}] {case['query'][:40]:40s} "
              f"→ {str(predicted):5s} (conf={confidence:.2f})")

    # 只统计有明确答案的
    definite = [r for r in results if r["correct"] is not None]
    accuracy = sum(r["correct"] for r in definite) / len(definite) if definite else 0.0
    total_time = round(time.time() - t0, 2)

    return {
        "accuracy":    round(accuracy, 4),
        "n_cases":     len(cases),
        "n_evaluated": len(definite),
        "total_time_s": total_time,
        "per_case":    results,
    }


# ── 评测2：蛋白注释 API 可达性 ────────────────────────────────────────────────
def eval_annotation_api() -> dict:
    from agents.annotation_agent import annotate_proteins

    test_proteins = ["APOB", "PCSK9", "CRP"]
    t0 = time.time()
    results = []

    try:
        ann = annotate_proteins(test_proteins)
        for r in ann:
            ok = bool(r.get("ensembl_id")) and r.get("top_disease", "—") != "—"
            results.append({
                "protein":       r["protein"],
                "ensembl_id":    r.get("ensembl_id", ""),
                "top_disease":   r.get("top_disease", ""),
                "druggable":     r.get("druggable", False),
                "gwas_snp_count":r.get("gwas_snp_count", 0),
                "ok":            ok,
            })
            print(f"  {'✓' if ok else '✗'} {r['protein']:8s} "
                  f"→ {r.get('top_disease','')[:40]:40s}  "
                  f"druggable={r.get('druggable')}  "
                  f"gwas_snps={r.get('gwas_snp_count')}")
        api_ok = all(r["ok"] for r in results)
    except Exception as e:
        api_ok = False
        results = [{"error": str(e)}]
        print(f"  ✗ annotation API 失败: {e}")

    return {
        "api_reachable": api_ok,
        "n_proteins":    len(test_proteins),
        "latency_s":     round(time.time() - t0, 2),
        "per_protein":   results,
    }


# ── 评测3：仿真引擎健全性 ─────────────────────────────────────────────────────
def eval_engine() -> dict:
    from engine_api import run_pwas_simulation

    t0 = time.time()
    try:
        res = run_pwas_simulation({"n_reps": 5, "use_real_ld": False,
                                   "use_metacycle": False})
        c = res["circadian_delta_auc"]
        n = res["null_delta_auc"]
        r = res["random_delta_auc"]

        # 基本健全性检查
        checks = {
            "circadian_delta_auc_is_float": isinstance(c, float),
            "null_delta_auc_near_zero":     abs(n) < 0.15,
            "top_proteins_nonempty":        len(res.get("top_proteins", [])) > 0,
            "corr_r2_snr_positive":         res.get("corr_r2_snr", 0) > 0,
        }
        all_ok = all(checks.values())

        print(f"  circadian ΔAuC = {c:+.4f}")
        print(f"  null     ΔAuC = {n:+.4f}")
        print(f"  random   ΔAuC = {r:+.4f}")
        for k, v in checks.items():
            print(f"  {'✓' if v else '✗'} {k}")

        return {
            "engine_ok":   all_ok,
            "delta_aucs":  {"circadian": c, "null": n, "random": r},
            "checks":      checks,
            "latency_s":   round(time.time() - t0, 2),
        }
    except Exception as e:
        print(f"  ✗ 引擎异常: {e}")
        return {"engine_ok": False, "error": str(e),
                "latency_s": round(time.time() - t0, 2)}


# ── 主程序 ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="使用真实 API（需 ANTHROPIC_API_KEY）")
    ap.add_argument("--skip-engine", action="store_true",
                    help="跳过引擎评测（节省时间）")
    args = ap.parse_args()

    print(f"\n{'='*60}")
    print(f"PWAS Agent 评测   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模式: {'真实 API' if args.full else 'Mock（无需 API key）'}")
    print(f"{'='*60}\n")

    report = {"timestamp": datetime.now().isoformat(), "mode": "full" if args.full else "mock"}

    # 1. 分类评测
    print("【1/3】性状节律性分类")
    clf = eval_classification(full=args.full)
    report["classification"] = clf
    print(f"  → 准确率 {clf['accuracy']:.1%}  ({clf['n_evaluated']}/{clf['n_cases']} 有答案案例)"
          f"  耗时 {clf['total_time_s']}s\n")

    # 2. 注释 API
    print("【2/3】蛋白注释 API")
    ann = eval_annotation_api()
    report["annotation"] = ann
    print(f"  → API {'可达' if ann['api_reachable'] else '不可达'}  "
          f"耗时 {ann['latency_s']}s\n")

    # 3. 引擎
    if not args.skip_engine:
        print("【3/3】仿真引擎健全性")
        eng = eval_engine()
        report["engine"] = eng
        print(f"  → 引擎 {'正常' if eng['engine_ok'] else '异常'}  "
              f"耗时 {eng['latency_s']}s\n")
    else:
        report["engine"] = {"skipped": True}

    # 汇总
    all_ok = (clf["accuracy"] >= 0.7
              and ann["api_reachable"]
              and report["engine"].get("engine_ok", True))

    print(f"{'='*60}")
    print(f"总体结果: {'✅ PASS' if all_ok else '❌ FAIL'}")
    print(f"  分类准确率: {clf['accuracy']:.1%}  (阈值 ≥ 70%)")
    print(f"  注释 API:  {'✓' if ann['api_reachable'] else '✗'}")
    if not args.skip_engine:
        print(f"  引擎健全:  {'✓' if report['engine'].get('engine_ok') else '✗'}")
    print(f"{'='*60}\n")

    report["overall_pass"] = all_ok
    with open(RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"结果已保存 → {RESULTS_PATH}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
