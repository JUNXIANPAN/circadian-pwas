"""
节律加权 PWAS 分析平台
用大白话提问，AI 帮你判断性状是否节律相关、跑模拟、注释蛋白、生成报告。
"""

import sys
import os
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ── 路径 ─────────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "scripts", "simulation3"))
sys.path.insert(0, os.path.join(_ROOT, "app"))


# ── 展示函数（必须在调用前定义）─────────────────────────────────────────────
def _show_literature(lit: dict):
    st.subheader("📚 文献检索结果")
    cols = st.columns(3)
    cols[0].metric("性状节律性", "是" if lit["is_circadian"] else "否")
    cols[1].metric("置信度", f"{lit.get('confidence', 0):.0%}")
    cols[2].metric("支持文献数", len(lit.get("evidence", [])))
    if lit.get("evidence"):
        with st.expander("查看引用文献（已验证 PMID）"):
            for e in lit["evidence"]:
                st.markdown(f"- **PMID {e['pmid']}** ({e.get('year','')})  \n  {e['title']}")


def _show_simulation(res: dict, is_circadian: bool = True):
    st.subheader("🔬 模拟结果")

    c_auc = res["circadian_delta_auc"]
    n_auc = res["null_delta_auc"]
    r_auc = res["random_delta_auc"]

    # 根据文献Agent判断，高亮对应场景并给出解读
    if is_circadian:
        st.info(
            f"📌 **文献支持该性状具有节律性** → 关注 Circadian ΔAuC = **{c_auc:+.4f}**。"
            f"正值表示节律加权方法在该场景下优于传统 PWAS。"
        )
    else:
        st.warning(
            f"📌 **文献不支持该性状具有节律性** → 节律加权方法对此类性状可能无效，"
            f"参考 Random ΔAuC = **{r_auc:+.4f}**（预期为负或接近0）。"
        )

    cols = st.columns(3)
    # 高亮相关场景
    main_col = 0 if is_circadian else 2
    for i, (label, val, tip) in enumerate([
        ("Circadian ΔAuC", c_auc, "节律性状场景：正值 = 加权方法更好"),
        ("Null ΔAuC",      n_auc, "无信号场景：应接近 0"),
        ("Random ΔAuC",    r_auc, "随机因果场景：可能为负"),
    ]):
        delta = "← 当前性状预期场景" if i == main_col else None
        cols[i].metric(label, f"{val:+.4f}", delta=delta, help=tip)

    scenarios = ["Circadian", "Null", "Random"]
    values    = [c_auc, n_auc, r_auc]
    # 当前性状对应场景用实色，其余用浅色
    colors = []
    for i, v in enumerate(values):
        if i == main_col:
            colors.append("#2ecc71" if v > 0 else "#e74c3c")
        else:
            colors.append("#bdc3c7")
    fig = go.Figure(go.Bar(x=scenarios, y=values, marker_color=colors))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title="ΔAuC by Scenario（加权 - 传统）",
                      yaxis_title="ΔAuC", height=300)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "💡 三个场景的数值本身与你输入的性状名无关（模拟用合成数据验证方法本身）。"
        "文献Agent的节律性判断决定哪个场景与你的问题最相关。"
    )

    if res.get("top_proteins"):
        with st.expander("加权排名 Top 10 蛋白"):
            st.dataframe(
                pd.DataFrame({"蛋白": res["top_proteins"],
                              "排名": range(1, len(res["top_proteins"]) + 1)}),
                use_container_width=True, hide_index=True,
            )


def _show_annotation(ann: list):
    st.subheader("🎯 蛋白靶点注释")
    if ann:
        st.dataframe(pd.DataFrame(ann), use_container_width=True, hide_index=True)
    else:
        st.info("无注释结果。")


def _fallback_report(query: str, lit, sim) -> str:
    if sim is None:
        return "模拟未完成，无法生成报告。"
    c = sim["circadian_delta_auc"]
    direction = "提升" if c > 0.01 else ("基本持平" if c > -0.01 else "下降")
    return f"""
**问题**：{query}

**模拟结论**：节律加权 PWAS 相比传统 PWAS，在 circadian 场景下 ΔAuC = {c:+.4f}，方法性能**{direction}**。

- Null 场景 ΔAuC ≈ {sim['null_delta_auc']:+.4f}（接近 0，方法不产生假信号 ✓）
- Random 场景 ΔAuC ≈ {sim['random_delta_auc']:+.4f}（用错对象的代价，属正常现象）

**建议**：{'该性状节律加权有望提升 PWAS 功效。' if c > 0.005 else '该性状节律加权效果有限，建议检查性状节律证据。'}
"""


# ── 页面配置 ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="节律加权 PWAS", page_icon="🧬", layout="wide")
st.title("🧬 节律加权 PWAS 分析平台")
st.caption("输入一个性状，AI 判断是否节律相关，并运行加权 PWAS 模拟分析。")

# ── 侧边栏 ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 模拟参数")
    n_reps   = st.slider("重复次数 (n_reps)", 5, 100, 20,
                         help="越多结果越稳定，但越慢。快速测试用 20。")
    tau      = st.slider("节律权重强度 (τ)", 0.1, 3.0, 1.0, step=0.1,
                         help="越大表示越强烈地偏向节律蛋白。")
    h2_med   = st.slider("蛋白介导遗传率 (h²_med)", 0.05, 0.50, 0.20, step=0.05,
                         help="性状遗传率中经蛋白传导的比例。")
    n_causal = st.slider("因果蛋白数", 5, 50, 20,
                         help="模拟中被设为因果蛋白的数量。")
    st.divider()
    st.caption("Agent 设置")
    check_literature = st.toggle("启用文献检索 Agent", value=True,
                                 help="联网检索文献，判断性状节律性并验证 PMID。")
    check_annotation = st.toggle("启用蛋白注释 Agent", value=True,
                                 help="查 Open Targets 注释命中蛋白的已知疾病关联。")

# ── 主界面 ────────────────────────────────────────────────────────────────────
query = st.text_area(
    "请用大白话描述你的问题",
    placeholder="例如：睡眠时长这个性状用节律加权 PWAS 会比传统方法好吗？",
    height=100,
)

col1, col2 = st.columns([1, 5])
with col1:
    run_btn = st.button("🚀 开始分析", type="primary", use_container_width=True)
with col2:
    st.caption("分析包含：文献检索 → 模拟 → 蛋白注释 → 报告生成，约需 1-3 分钟。")

# ── 分析流程 ──────────────────────────────────────────────────────────────────
if run_btn and query.strip():
    st.divider()

    # ① 文献 Agent
    lit_result = None
    if check_literature:
        with st.status("📚 文献 Agent：判断性状节律性...", expanded=True) as status:
            try:
                from agents.orchestrator import check_circadian_trait
                lit_result = check_circadian_trait(query)
                label = (f"✅ 节律性状（置信度 {lit_result['confidence']:.0%}）"
                         if lit_result["is_circadian"]
                         else f"⚠️ 可能非节律性状（置信度 {lit_result['confidence']:.0%}）")
                status.update(label=label, state="complete")
            except Exception as e:
                status.update(label=f"文献 Agent 不可用（{e}），跳过", state="error")
                lit_result = {"is_circadian": True, "confidence": 0.5,
                              "evidence": [], "trait_name": query}
        _show_literature(lit_result)

    # ② 仿真 Agent
    sim_result = None
    with st.status("🔬 仿真 Agent：运行节律加权 PWAS 模拟...", expanded=True) as status:
        try:
            from engine_api import run_pwas_simulation
            sim_result = run_pwas_simulation({
                "n_reps": n_reps, "tau": tau, "h2_med": h2_med,
                "n_causal": n_causal, "use_real_ld": False, "use_metacycle": False,
            })
            status.update(label="✅ 模拟完成", state="complete")
        except Exception as e:
            status.update(label=f"模拟失败：{e}", state="error")
    if sim_result:
        if sim_result.get("_unavailable"):
            st.warning(f"⚠️ {sim_result['_reason']}")
        else:
            is_circ = lit_result["is_circadian"] if lit_result else True
            _show_simulation(sim_result, is_circadian=is_circ)

    # ③ 蛋白注释 Agent
    ann_result = None
    if check_annotation and sim_result and sim_result.get("top_proteins"):
        with st.status("🔎 注释 Agent：查询 Open Targets...", expanded=True) as status:
            try:
                from agents.orchestrator import annotate_proteins
                ann_result = annotate_proteins(sim_result["top_proteins"])
                status.update(label="✅ 注释完成", state="complete")
            except Exception as e:
                status.update(label=f"注释 Agent 不可用（{e}），跳过", state="error")
        if ann_result:
            _show_annotation(ann_result)

    # ④ 报告 Agent
    with st.status("📝 报告 Agent：生成大白话报告...", expanded=True) as status:
        try:
            from agents.orchestrator import generate_report
            report = generate_report(query, lit_result, sim_result, ann_result)
        except Exception as e:
            report = _fallback_report(query, lit_result, sim_result)
        status.update(label="✅ 报告生成完毕", state="complete")

    st.divider()
    st.subheader("📋 分析报告")
    st.markdown(report)

elif run_btn:
    st.warning("请先输入问题。")
