# PWAS 项目说明

## 项目概述
节律加权 PWAS（Proteome-Wide Association Study）——用蛋白质昼夜节律性（R²）对 PWAS 分数加权，提升节律相关性状的因果蛋白发现能力。

## 核心引擎
- **主文件**：`scripts/simulation3/simulation3_4.py`（~1400行）
- **API封装**：`scripts/simulation3/engine_api.py` → `run_pwas_simulation(params: dict) -> dict`
- **Web前端**：`app/streamlit_app.py`
- **Agent层**：`app/agents/orchestrator.py`（总控）+ 各子Agent

## 运行方式
```bash
# SLURM 批量运行（200次重复）
sbatch submit_simulation3_4.sh

# 本地快速测试（20次重复）
conda activate pwas_env
python -c "from scripts.simulation3.engine_api import run_pwas_simulation; print(run_pwas_simulation({'n_reps':5}))"

# 启动 Web 前端
cd /data2/pan/pwas
streamlit run app/streamlit_app.py
```

## 关键参数
| 参数 | 默认值 | 含义 |
|------|--------|------|
| n_reps | 200 | 模拟重复次数 |
| tau | 1.0 | 节律权重强度 |
| h2_med | 0.20 | 蛋白介导遗传率 |
| h2_non_med | 0.10 | 直接遗传率（绕过蛋白） |
| n_causal | 20 | 每次重复的因果蛋白数 |

## 数据路径
- 蛋白节律文件：`raw_data/circadian_info/report.pg_matrix.tsv`
- pQTL 数据：`work/pqtl_topk.csv`
- MAF 缓存：`raw_data/reference/ensembl_maf.csv`（1750 SNPs, gnomAD-NFE）
- UKBB-LD：`/data/CommonData/ukbb-ld/`（2763个块，各0.6-1.9GB，加载慢）

## 重要设计决策
- **因果蛋白选取**：用真实 SNR=A/σ（非R²），避免循环
- **UKBB-LD加载**：用 `mmap_mode='r'` 避免OOM；节点内存需≥64GB
- **R²**：MetaCycle/cosinor 对 pop_mean（P×8矩阵）拟合，含测量噪声
- **三场景**：circadian（SNR偏向）/ random（随机）/ null（y=噪声）

## Agent 环境变量
```
ANTHROPIC_API_KEY=...   # Claude API（可选，无则用Mock模式）
NCBI_API_KEY=...        # PubMed E-utilities 加速（可选）
```

## 待完成
- [ ] Step 4: 文献 Agent（literature_agent.py）+ PMID 验证
- [ ] Step 5: 蛋白注释 Agent（annotation_agent.py）
- [ ] Step 6: eval框架 + CI + 部署
