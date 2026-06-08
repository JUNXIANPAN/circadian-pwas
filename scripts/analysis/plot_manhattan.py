import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("raw_data/gwas/gwas_for_plot.txt", sep="\t")

# 转换类型
df['CHR'] = df['CHR'].astype(str).str.replace('chr', '', regex=False)
df['CHR'] = df['CHR'].replace({'X': 23, 'Y': 24, 'MT': 25, 'M': 25})
df['CHR'] = pd.to_numeric(df['CHR'], errors='coerce')
df = df.dropna(subset=['CHR'])

df['CHR'] = df['CHR'].astype(int)
df['BP'] = df['BP'].astype(int)
df['P'] = df['P'].astype(float)

# 排序
df = df.sort_values(['CHR', 'BP'])

# 计算 -log10(p)
df['logP'] = -np.log10(df['P'])

# 生成 cumulative position（关键！）
df['ind'] = range(len(df))
df_grouped = df.groupby('CHR')

# 画图
plt.figure(figsize=(12,6))

colors = ['black', 'gray']
x_labels = []
x_labels_pos = []

for i, (name, group) in enumerate(df_grouped):
    group.plot(kind='scatter',
               x='ind',
               y='logP',
               color=colors[i % 2],
               s=1,
               ax=plt.gca())

    x_labels.append(name)
    x_labels_pos.append((group['ind'].iloc[-1] + group['ind'].iloc[0]) / 2)

# GWAS threshold
plt.axhline(-np.log10(5e-8), color='red', linestyle='--')

# 坐标轴
plt.xticks(x_labels_pos, x_labels)
plt.xlabel('Chromosome')
plt.ylabel('-log10(P)')
plt.title('Manhattan Plot')

plt.tight_layout()
plt.savefig("manhattan.png", dpi=300)
plt.show()