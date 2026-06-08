import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 你的数据（直接粘贴）
data = """
circadian_mediation,RP,200,0.08414455782312925,0.016628323321579555,0.0011758000180451319
circadian_mediation,full,200,0.08471451247165533,0.017338738760486045,0.0012260339754761715
circadian_mediation,uniform,200,0.0,0.0,0.0
non_circadian_mediation,RP,200,-0.007541950113378684,0.01181397333125235,0.0008353740655285564
non_circadian_mediation,full,200,-0.004910544217687068,0.008802757285772438,0.0006224489369908978
non_circadian_mediation,uniform,200,0.0,0.0,0.0
wrong_phase,RP,200,-0.11973571428571429,0.018669309180277305,0.0013201195121442348
wrong_phase,full,200,-0.11957052154195011,0.018563873205514593,0.0013126640628706619
wrong_phase,uniform,200,0.0,0.0,0.0
"""

# 读入数据
from io import StringIO
df = pd.read_csv(StringIO(data), 
                 names=['scenario','prior','n','mean_delta_auc','sd_delta_auc','se_delta_auc'])

# 去掉 null 场景（n=0 的行）
df = df[df['n'] > 0]

# 设置图形
plt.figure(figsize=(8, 5))
scenarios = df['scenario'].unique()
priors = df['prior'].unique()
x = np.arange(len(scenarios))          # scenario 位置
width = 0.25                           # 条宽
colors = {'RP':'steelblue', 'full':'orange', 'uniform':'gray'}

# 为每个 prior 画一组条形
for i, prior in enumerate(priors):
    subset = df[df['prior'] == prior]
    means = [subset[subset['scenario']==s]['mean_delta_auc'].values[0] for s in scenarios]
    errors = [subset[subset['scenario']==s]['se_delta_auc'].values[0] for s in scenarios]
    plt.bar(x + i*width, means, width, 
            label=prior, color=colors[prior],
            yerr=errors, capsize=3, error_kw={'elinewidth':1, 'markeredgewidth':1})

# 装饰
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
plt.xticks(x + width, scenarios, rotation=15, ha='right')
plt.ylabel('Mean ΔAUC')
plt.title('Mean ΔAUC by Scenario and Prior (error bars = ±SE)')
plt.legend(title='Prior')
plt.tight_layout()
plt.show()