# -*- coding: utf-8 -*-
"""
问题2: 高级科研风格可视化 (fig13-fig20)
- 小提琴图、桑基图、日历热力图、平行坐标图
- 相关性矩阵、RFM分析、时间序列模式、用户分层金字塔
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
import os
from scipy import stats

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
DATA_DIR = r'g:\A.lsit\08.建模\校赛-B 题  销售商品行为预测\B题：附件1\赛题数据'
OUTPUT_DIR = r'g:\A.lsit\08.建模\校赛-B 题  销售商品行为预测\问题2'

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
})

COLORS = {
    'primary': '#2C3E50',
    'secondary': '#3498DB',
    'accent': '#E74C3C',
    'palette': ['#2C3E50', '#3498DB', '#E74C3C', '#27AE60', '#F39C12', '#8E44AD', '#16A085', '#D35400'],
}

def save_fig(fig, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] {filename}")

# ==================== 加载数据 ====================
print("加载数据...")
customers = pd.read_csv(os.path.join(DATA_DIR, 'customers_info.csv'))
behaviors = pd.read_csv(os.path.join(DATA_DIR, 'user_behavior.csv'))
products = pd.read_csv(os.path.join(DATA_DIR, 'products.csv'))

customers.columns = ['用户ID', '用户所在地', '年龄', '性别', '权益']
behaviors.columns = ['用户ID', '商品ID', '时间', '行为', '备注']
products.columns = ['商品ID', '商品类型', '具体商品', '成本', '单价', '库存']

behaviors['时间'] = pd.to_datetime(behaviors['时间'], errors='coerce')
customers['性别'] = customers['性别'].fillna('未知')
customers['年龄'] = customers['年龄'].fillna(customers['年龄'].median())
products['单价'] = products['单价'].fillna(products['单价'].median())
behaviors = behaviors.dropna(subset=['时间'])
behaviors = behaviors[behaviors['行为'].isin(['浏览', '收藏', '加购', '购买'])]

# 合并数据
df = behaviors.merge(customers, on='用户ID', how='left')
df = df.merge(products, on='商品ID', how='left')
df['月份'] = df['时间'].dt.month
df['星期'] = df['时间'].dt.dayofweek
df['小时'] = df['时间'].dt.hour
df['日期'] = df['时间'].dt.date
df['周末'] = df['时间'].dt.dayofweek >= 5

# 构建用户画像
behavior_counts = df.groupby(['用户ID', '行为']).size().unstack(fill_value=0)
behavior_counts.columns = [f'{col}_次数' for col in behavior_counts.columns]

purchase_df = df[df['行为'] == '购买']
purchase_stats = purchase_df.groupby('用户ID').agg({
    '单价': ['count', 'sum', 'mean'],
    '商品ID': 'nunique',
}).reset_index()
purchase_stats.columns = ['用户ID', '购买次数', '总消费金额', '客单价', '购买商品种类数']

user_dates = df.groupby('用户ID')['日期'].nunique().reset_index()
user_dates.columns = ['用户ID', '活跃天数']

user_first_last = df.groupby('用户ID')['时间'].agg(['min', 'max']).reset_index()
user_first_last.columns = ['用户ID', '首次活跃时间', '最后活跃时间']
user_first_last['留存天数'] = (user_first_last['最后活跃时间'] - user_first_last['首次活跃时间']).dt.days

user_profile = behavior_counts.reset_index()
user_profile = user_profile.merge(user_dates, on='用户ID', how='outer')
user_profile = user_profile.merge(purchase_stats, on='用户ID', how='outer')
user_profile = user_profile.merge(user_first_last[['用户ID', '留存天数']], on='用户ID', how='outer')
user_profile = user_profile.merge(customers[['用户ID', '年龄', '性别', '权益']], on='用户ID', how='left')

for col in ['浏览_次数', '收藏_次数', '加购_次数', '购买_次数']:
    if col not in user_profile.columns:
        user_profile[col] = 0
    user_profile[col] = user_profile[col].fillna(0)
user_profile['活跃天数'] = user_profile['活跃天数'].fillna(0)
user_profile['购买次数'] = user_profile['购买次数'].fillna(0)
user_profile['总消费金额'] = user_profile['总消费金额'].fillna(0)
user_profile['客单价'] = user_profile['客单价'].fillna(0)
user_profile['购买商品种类数'] = user_profile['购买商品种类数'].fillna(0)
user_profile['留存天数'] = user_profile['留存天数'].fillna(0)

user_profile['浏览到收藏转化率'] = user_profile.apply(
    lambda x: x['收藏_次数'] / x['浏览_次数'] if x['浏览_次数'] > 0 else 0, axis=1)
user_profile['收藏到加购转化率'] = user_profile.apply(
    lambda x: x['加购_次数'] / x['收藏_次数'] if x['收藏_次数'] > 0 else 0, axis=1)
user_profile['加购到购买转化率'] = user_profile.apply(
    lambda x: x['购买_次数'] / x['加购_次数'] if x['加购_次数'] > 0 else 0, axis=1)
user_profile['整体转化率'] = user_profile.apply(
    lambda x: x['购买_次数'] / x['浏览_次数'] if x['浏览_次数'] > 0 else 0, axis=1)

# 聚类
cluster_features = ['浏览_次数', '收藏_次数', '加购_次数', '购买_次数', 
                    '活跃天数', '总消费金额', '客单价', '留存天数',
                    '浏览到收藏转化率', '收藏到加购转化率', '加购到购买转化率']
cluster_df = user_profile[cluster_features].copy()
cluster_df = cluster_df.replace([np.inf, -np.inf], np.nan).fillna(0)
scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_df)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10, max_iter=300)
user_profile['行为模式'] = kmeans.fit_predict(cluster_scaled)

print(f"用户数: {len(user_profile)}, 聚类数: 2")

# ==================== 高级可视化 ====================
print("\n生成高级科研风格可视化...")

# 图13: 用户行为指标小提琴图 (按行为模式分组)
print("\n生成图13: 行为指标小提琴图...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

metrics = ['浏览_次数', '收藏_次数', '加购_次数', '购买_次数', '活跃天数', '留存天数']
labels = ['浏览次数', '收藏次数', '加购次数', '购买次数', '活跃天数', '留存天数']

for i, (col, label) in enumerate(zip(metrics, labels)):
    ax = axes[i]
    data = user_profile[[col, '行为模式']].copy()
    data[col] = data[col].clip(upper=data[col].quantile(0.95))
    
    parts = ax.violinplot(
        [data[data['行为模式'] == 0][col].values, data[data['行为模式'] == 1][col].values],
        positions=[1, 2], widths=0.6, showmeans=True, showextrema=True, showmedians=True
    )
    
    for pc, color in zip(parts['bodies'], [COLORS['palette'][0], COLORS['palette'][1]]):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
        if partname in parts:
            parts[partname].set_edgecolor('black')
            parts[partname].set_linewidth(1.5)
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['模式0\n(低频用户)', '模式1\n(高频用户)'], fontsize=10)
    ax.set_ylabel(label, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.15, axis='y', linestyle='--')
    
    g0 = data[data['行为模式'] == 0][col]
    g1 = data[data['行为模式'] == 1][col]
    if len(g0) > 0 and len(g1) > 0:
        _, p_val = stats.mannwhitneyu(g0, g1)
        y_max = max(data[col].max(), 0)
        ax.text(1.5, y_max * 0.95, f'p<{0.001 if p_val < 0.001 else p_val:.3f}', 
                ha='center', va='top', fontsize=9, fontweight='bold', color=COLORS['accent'])

axes[5].set_visible(False)
fig.suptitle('用户行为指标分布对比 (小提琴图)', fontsize=14, fontweight='bold', y=1.02)
fig.text(0.02, 0.02, '数据来源：电商平台用户行为数据 | 注：已去除5%极端值 | 标注为Mann-Whitney U检验p值', 
         fontsize=8, style='italic', alpha=0.6, transform=fig.transFigure)
save_fig(fig, 'fig13_violin_plots.png')

# 图14: 行为序列桑基图
print("生成图14: 行为序列桑基图...")
fig, ax = plt.subplots(figsize=(14, 8))

# 计算各行为类型的总次数
behavior_total = df.groupby('行为').size()
stage_counts = {
    '浏览': behavior_total.get('浏览', 0),
    '收藏': behavior_total.get('收藏', 0),
    '加购': behavior_total.get('加购', 0),
    '购买': behavior_total.get('购买', 0),
}

# 计算行为间的流转次数（基于用户ID）
behavior_flow = df.groupby(['用户ID', '行为']).size().unstack(fill_value=0)
for col in ['浏览', '收藏', '加购', '购买']:
    if col not in behavior_flow.columns:
        behavior_flow[col] = 0

flow_data = {
    ('浏览', '收藏'): len(behavior_flow[(behavior_flow['浏览'] > 0) & (behavior_flow['收藏'] > 0)]),
    ('浏览', '加购'): len(behavior_flow[(behavior_flow['浏览'] > 0) & (behavior_flow['加购'] > 0)]),
    ('浏览', '购买'): len(behavior_flow[(behavior_flow['浏览'] > 0) & (behavior_flow['购买'] > 0)]),
    ('收藏', '加购'): len(behavior_flow[(behavior_flow['收藏'] > 0) & (behavior_flow['加购'] > 0)]),
    ('收藏', '购买'): len(behavior_flow[(behavior_flow['收藏'] > 0) & (behavior_flow['购买'] > 0)]),
    ('加购', '购买'): len(behavior_flow[(behavior_flow['加购'] > 0) & (behavior_flow['购买'] > 0)]),
}

stages = ['浏览', '收藏', '加购', '购买']
stage_colors = [COLORS['palette'][0], COLORS['palette'][1], COLORS['palette'][2], COLORS['palette'][3]]
x_positions = [0.1, 0.35, 0.6, 0.85]

max_count = max(stage_counts.values())

for i, (stage, x) in enumerate(zip(stages, x_positions)):
    count = stage_counts[stage]
    height = count / max_count * 0.6
    
    rect = plt.Rectangle((x - 0.025, 0.5 - height/2), 0.05, height, 
                         facecolor=stage_colors[i], alpha=0.85, edgecolor='white', linewidth=2)
    ax.add_patch(rect)
    
    ax.text(x, 0.5 + height/2 + 0.04, f'{stage}', 
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.text(x, 0.5 - height/2 - 0.04, f'{count:,}', 
            ha='center', va='top', fontsize=10, fontweight='bold', color=stage_colors[i])

from matplotlib.path import Path
import matplotlib.patches as mpatches

max_flow = max(flow_data.values())

for (src, tgt), count in flow_data.items():
    src_idx = stages.index(src)
    tgt_idx = stages.index(tgt)
    
    src_x = x_positions[src_idx] + 0.025
    tgt_x = x_positions[tgt_idx] - 0.025
    
    src_height = stage_counts[src] / max_count * 0.6
    tgt_height = stage_counts[tgt] / max_count * 0.6
    
    src_y_start = 0.5 - src_height / 2
    tgt_y_start = 0.5 - tgt_height / 2
    
    ctrl_x = (src_x + tgt_x) / 2
    
    path_data = [
        (Path.MOVETO, (src_x, src_y_start)),
        (Path.CURVE4, (ctrl_x, src_y_start)),
        (Path.CURVE4, (ctrl_x, tgt_y_start)),
        (Path.CURVE4, (tgt_x, tgt_y_start)),
    ]
    codes, verts = zip(*path_data)
    path = Path(verts, codes)
    
    alpha = min(count / max_flow, 1) * 0.4 + 0.2
    lw = 1 + count / max_flow * 6
    patch = mpatches.PathPatch(path, facecolor='none', edgecolor=stage_colors[src_idx], 
                               linewidth=lw, alpha=alpha)
    ax.add_patch(patch)
    
    mid_x = (src_x + tgt_x) / 2
    mid_y = (src_y_start + tgt_y_start) / 2
    ax.text(mid_x, mid_y + 0.02, f'{count:,}', ha='center', va='bottom', 
            fontsize=8, fontweight='bold', color=stage_colors[src_idx])

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('用户行为序列流转图 (桑基图)', fontsize=14, fontweight='bold', pad=15)
ax.axis('off')
fig.text(0.02, 0.02, '数据来源：电商平台用户行为数据 | 节点高度表示行为总次数 | 线条粗细表示流转用户数量', 
         fontsize=8, style='italic', alpha=0.6, transform=fig.transFigure)
save_fig(fig, 'fig14_sankey_flow.png')

# 图15: 用户行为日历热力图
print("生成图15: 行为日历热力图...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax = axes[0, 0]
month_hour = df.groupby(['月份', '小时']).size().unstack(fill_value=0)
sns.heatmap(month_hour, ax=ax, cmap='YlOrRd', cbar_kws={'label': '行为次数'},
            linewidths=0.3, linecolor='white', annot=False)
ax.set_xlabel('小时', fontsize=11, fontweight='bold')
ax.set_ylabel('月份', fontsize=11, fontweight='bold')
ax.set_title('月份×小时行为热力图', fontsize=12, fontweight='bold', pad=10)

ax = axes[0, 1]
weekday_hour = df.groupby(['星期', '小时']).size().unstack(fill_value=0)
weekday_labels = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
weekday_hour.index = [weekday_labels[i] for i in weekday_hour.index]
sns.heatmap(weekday_hour, ax=ax, cmap='YlOrRd', cbar_kws={'label': '行为次数'},
            linewidths=0.3, linecolor='white', annot=False)
ax.set_xlabel('小时', fontsize=11, fontweight='bold')
ax.set_ylabel('星期', fontsize=11, fontweight='bold')
ax.set_title('星期×小时行为热力图', fontsize=12, fontweight='bold', pad=10)

ax = axes[1, 0]
month_weekday = df.groupby(['月份', '星期']).size().unstack(fill_value=0)
month_weekday.index = [f'{m}月' for m in month_weekday.index]
month_weekday.columns = weekday_labels
sns.heatmap(month_weekday, ax=ax, cmap='YlOrRd', cbar_kws={'label': '行为次数'},
            linewidths=0.3, linecolor='white', annot=False)
ax.set_xlabel('星期', fontsize=11, fontweight='bold')
ax.set_ylabel('月份', fontsize=11, fontweight='bold')
ax.set_title('月份×星期行为热力图', fontsize=12, fontweight='bold', pad=10)

ax = axes[1, 1]
behavior_hour = df.groupby(['行为', '小时']).size().unstack(fill_value=0)
behavior_order = ['浏览', '收藏', '加购', '购买']
behavior_hour = behavior_hour.reindex(behavior_order)
sns.heatmap(behavior_hour, ax=ax, cmap='YlOrRd', cbar_kws={'label': '行为次数'},
            linewidths=0.3, linecolor='white', annot=False)
ax.set_xlabel('小时', fontsize=11, fontweight='bold')
ax.set_ylabel('行为类型', fontsize=11, fontweight='bold')
ax.set_title('行为类型×小时热力图', fontsize=12, fontweight='bold', pad=10)

fig.suptitle('用户行为时间模式多维热力图', fontsize=14, fontweight='bold', y=1.02)
fig.text(0.02, 0.02, '数据来源：电商平台用户行为数据', 
         fontsize=8, style='italic', alpha=0.6, transform=fig.transFigure)
save_fig(fig, 'fig15_calendar_heatmap.png')

# 图16: 平行坐标图
print("生成图16: 平行坐标图...")
fig, ax = plt.subplots(figsize=(14, 8))

parallel_features = ['浏览_次数', '收藏_次数', '加购_次数', '购买_次数', 
                     '活跃天数', '客单价', '留存天数']
parallel_labels = ['浏览次数', '收藏次数', '加购次数', '购买次数', 
                   '活跃天数', '客单价(元)', '留存天数']

parallel_data = user_profile[parallel_features + ['行为模式']].copy()
parallel_data = parallel_data.replace([np.inf, -np.inf], np.nan).fillna(0)

for col in parallel_features:
    min_val = parallel_data[col].min()
    max_val = parallel_data[col].max()
    if max_val > min_val:
        parallel_data[col] = (parallel_data[col] - min_val) / (max_val - min_val)

sample_size = min(2000, len(parallel_data))
sample_idx = np.random.choice(len(parallel_data), sample_size, replace=False)
sample_data = parallel_data.iloc[sample_idx]

colors_map = {0: COLORS['palette'][0], 1: COLORS['palette'][1]}
alpha_map = {0: 0.15, 1: 0.15}

for idx, row in sample_data.iterrows():
    cluster = int(row['行为模式'])
    y_values = [row[col] for col in parallel_features]
    ax.plot(range(len(parallel_features)), y_values, 
            color=colors_map[cluster], alpha=alpha_map[cluster], linewidth=0.8)

for cluster in [0, 1]:
    cluster_data = parallel_data[parallel_data['行为模式'] == cluster][parallel_features]
    means = cluster_data.mean()
    ax.plot(range(len(parallel_features)), means.values, 
            color=colors_map[cluster], linewidth=3, alpha=0.9,
            label=f'模式{cluster}均值', marker='o', markersize=8)

ax.set_xticks(range(len(parallel_features)))
ax.set_xticklabels(parallel_labels, fontsize=10, rotation=45, ha='right')
ax.set_ylabel('归一化值', fontsize=12, fontweight='bold')
ax.set_title('用户行为指标平行坐标图', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11, framealpha=0.9)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.2, axis='y', linestyle='--')
fig.text(0.02, 0.02, f'数据来源：电商平台用户行为数据 | 细线: {sample_size}个采样用户 | 粗线: 各模式均值', 
         fontsize=8, style='italic', alpha=0.6, transform=fig.transFigure)
save_fig(fig, 'fig16_parallel_coordinates.png')

# 图17: 用户行为指标相关性矩阵图
print("生成图17: 相关性矩阵图...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax = axes[0]
corr_features = ['浏览_次数', '收藏_次数', '加购_次数', '购买_次数', 
                 '活跃天数', '总消费金额', '客单价', '留存天数',
                 '浏览到收藏转化率', '收藏到加购转化率', '加购到购买转化率', '整体转化率']
corr_labels = ['浏览次数', '收藏次数', '加购次数', '购买次数', 
               '活跃天数', '总消费金额', '客单价', '留存天数',
               '浏览→收藏', '收藏→加购', '加购→购买', '整体转化率']

corr_matrix = user_profile[corr_features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, ax=ax, mask=mask, cmap='RdBu_r', annot=True, fmt='.2f',
            cbar_kws={'label': '相关系数'}, linewidths=0.5, linecolor='white',
            square=True, vmin=-1, vmax=1)
ax.set_title('用户行为指标相关性矩阵', fontsize=13, fontweight='bold', pad=12)
ax.set_xticklabels(corr_labels, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(corr_labels, fontsize=9)

ax = axes[1]
scatter_features = ['活跃天数', '购买_次数', '总消费金额', '客单价']
g = sns.pairplot(user_profile[scatter_features + ['行为模式']].sample(min(3000, len(user_profile))),
                 hue='行为模式', palette=[COLORS['palette'][0], COLORS['palette'][1]],
                 diag_kind='kde', plot_kws={'alpha': 0.5, 's': 20},
                 diag_kws={'fill': True, 'alpha': 0.5})
g.fig.suptitle('关键指标散点图矩阵', fontsize=13, fontweight='bold', y=1.02)
g.fig.tight_layout()
g.savefig(os.path.join(OUTPUT_DIR, 'fig17b_scatter_matrix.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close(g.fig)

ax.text(0.5, 0.5, '散点图矩阵已单独保存为 fig17b_scatter_matrix.png', 
        ha='center', va='center', fontsize=12, transform=ax.transAxes)
ax.set_title('关键指标散点图矩阵', fontsize=13, fontweight='bold', pad=12)
ax.axis('off')

fig.suptitle('用户行为指标相关性分析', fontsize=14, fontweight='bold', y=1.02)
fig.text(0.02, 0.02, '数据来源：电商平台用户行为数据', 
         fontsize=8, style='italic', alpha=0.6, transform=fig.transFigure)
save_fig(fig, 'fig17_correlation_analysis.png')

# 图18: 用户价值RFM分析
print("生成图18: RFM用户价值分析...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

rfm_data = purchase_df.groupby('用户ID').agg({
    '时间': ['min', 'max', 'count'],
    '单价': 'sum'
}).reset_index()
rfm_data.columns = ['用户ID', '首次购买时间', '最后购买时间', '购买频次', '消费金额']
rfm_data['最近购买天数'] = (df['时间'].max() - rfm_data['最后购买时间']).dt.days

ax = axes[0, 0]
scatter = ax.scatter(rfm_data['最近购买天数'], rfm_data['购买频次'], 
                     c=rfm_data['消费金额'], cmap='YlOrRd', 
                     alpha=0.5, s=30, edgecolors='white', linewidth=0.5)
ax.set_xlabel('最近购买天数 (R)', fontsize=11, fontweight='bold')
ax.set_ylabel('购买频次 (F)', fontsize=11, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('消费金额 (M)', fontsize=10, fontweight='bold')
ax.set_title('RFM三维用户价值分布', fontsize=12, fontweight='bold', pad=12)
ax.grid(True, alpha=0.15, linestyle='--')

ax = axes[0, 1]
ax.hist(rfm_data['最近购买天数'], bins=50, color=COLORS['palette'][0], 
        alpha=0.7, edgecolor='white', linewidth=1)
ax.axvline(x=rfm_data['最近购买天数'].median(), color=COLORS['accent'], 
           linestyle='--', linewidth=2, label=f'中位数: {rfm_data["最近购买天数"].median():.0f}天')
ax.set_xlabel('最近购买天数', fontsize=11, fontweight='bold')
ax.set_ylabel('用户数', fontsize=11, fontweight='bold')
ax.set_title('R-最近购买天数分布', fontsize=12, fontweight='bold', pad=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.15, axis='y', linestyle='--')

ax = axes[1, 0]
ax.hist(rfm_data['购买频次'], bins=50, color=COLORS['palette'][1], 
        alpha=0.7, edgecolor='white', linewidth=1)
ax.axvline(x=rfm_data['购买频次'].median(), color=COLORS['accent'], 
           linestyle='--', linewidth=2, label=f'中位数: {rfm_data["购买频次"].median():.0f}次')
ax.set_xlabel('购买频次', fontsize=11, fontweight='bold')
ax.set_ylabel('用户数', fontsize=11, fontweight='bold')
ax.set_title('F-购买频次分布', fontsize=12, fontweight='bold', pad=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.15, axis='y', linestyle='--')

ax = axes[1, 1]
ax.hist(rfm_data['消费金额'], bins=50, color=COLORS['palette'][2], 
        alpha=0.7, edgecolor='white', linewidth=1)
ax.axvline(x=rfm_data['消费金额'].median(), color=COLORS['accent'], 
           linestyle='--', linewidth=2, label=f'中位数: {rfm_data["消费金额"].median():.0f}元')
ax.set_xlabel('消费金额 (元)', fontsize=11, fontweight='bold')
ax.set_ylabel('用户数', fontsize=11, fontweight='bold')
ax.set_title('M-消费金额分布', fontsize=12, fontweight='bold', pad=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.15, axis='y', linestyle='--')

fig.suptitle('RFM用户价值分析模型', fontsize=14, fontweight='bold', y=1.02)
fig.text(0.02, 0.02, '数据来源：电商平台用户行为数据 | R:最近购买时间 F:购买频次 M:消费金额', 
         fontsize=8, style='italic', alpha=0.6, transform=fig.transFigure)
save_fig(fig, 'fig18_rfm_analysis.png')

# 图19: 用户行为时间序列模式
print("生成图19: 用户行为时间序列模式...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

ax = axes[0, 0]
monthly_behavior = df.groupby(['月份', '行为']).size().unstack(fill_value=0)
for col in ['浏览', '收藏', '加购', '购买']:
    if col in monthly_behavior.columns:
        ax.plot(monthly_behavior.index, monthly_behavior[col], 'o-', 
                linewidth=2, markersize=6, label=col)
ax.set_xlabel('月份', fontsize=11, fontweight='bold')
ax.set_ylabel('行为次数', fontsize=11, fontweight='bold')
ax.set_title('各行为类型月度趋势', fontsize=12, fontweight='bold', pad=12)
ax.set_xticks(range(1, 13))
ax.set_xticklabels([f'{m}月' for m in range(1, 13)])
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.15, linestyle='--')

ax = axes[0, 1]
hourly_behavior = df.groupby('小时').size()
ax.plot(hourly_behavior.index, hourly_behavior.values, 'o-', 
        linewidth=2.5, markersize=6, color=COLORS['primary'])
ax.fill_between(hourly_behavior.index, hourly_behavior.values, alpha=0.2, color=COLORS['primary'])
ax.set_xlabel('小时', fontsize=11, fontweight='bold')
ax.set_ylabel('行为次数', fontsize=11, fontweight='bold')
ax.set_title('24小时行为分布', fontsize=12, fontweight='bold', pad=12)
ax.set_xticks(range(0, 24, 2))
ax.grid(True, alpha=0.15, linestyle='--')

ax = axes[1, 0]
weekend_data = df.groupby('周末').size()
weekend_labels = ['工作日', '周末']
bars = ax.bar(weekend_labels, weekend_data.values, 
              color=[COLORS['palette'][1], COLORS['palette'][2]], 
              alpha=0.85, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, weekend_data.values):
    ax.annotate(f'{val:,}',
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 8), textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('行为次数', fontsize=11, fontweight='bold')
ax.set_title('周末vs工作日行为对比', fontsize=12, fontweight='bold', pad=12)
ax.grid(True, alpha=0.15, axis='y', linestyle='--')

ax = axes[1, 1]
behavior_weekday = df.groupby(['行为', '星期']).size().unstack(fill_value=0)
behavior_order = ['浏览', '收藏', '加购', '购买']
behavior_weekday = behavior_weekday.reindex(behavior_order)
behavior_weekday.columns = [f'周{i+1}' if i < 5 else ('周六' if i == 5 else '周日') for i in behavior_weekday.columns]
sns.heatmap(behavior_weekday, ax=ax, cmap='YlOrRd', annot=True, fmt='d',
            cbar_kws={'label': '行为次数'}, linewidths=0.5, linecolor='white')
ax.set_xlabel('星期', fontsize=11, fontweight='bold')
ax.set_ylabel('行为类型', fontsize=11, fontweight='bold')
ax.set_title('行为类型×星期分布', fontsize=12, fontweight='bold', pad=12)

fig.suptitle('用户行为时间模式分析', fontsize=14, fontweight='bold', y=1.02)
fig.text(0.02, 0.02, '数据来源：电商平台用户行为数据', 
         fontsize=8, style='italic', alpha=0.6, transform=fig.transFigure)
save_fig(fig, 'fig19_time_series_patterns.png')

# 图20: 用户分层金字塔
print("生成图20: 用户分层金字塔...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax = axes[0]
purchase_freq = user_profile['购买_次数']
layers = [
    ('核心用户', purchase_freq >= purchase_freq.quantile(0.9)),
    ('活跃用户', (purchase_freq >= purchase_freq.quantile(0.7)) & (purchase_freq < purchase_freq.quantile(0.9))),
    ('普通用户', (purchase_freq >= purchase_freq.quantile(0.5)) & (purchase_freq < purchase_freq.quantile(0.7))),
    ('低频用户', (purchase_freq >= purchase_freq.quantile(0.3)) & (purchase_freq < purchase_freq.quantile(0.5))),
    ('沉睡用户', purchase_freq < purchase_freq.quantile(0.3)),
]

layer_colors = ['#E74C3C', '#F39C12', '#3498DB', '#27AE60', '#95A5A6']
layer_counts = [l[1].sum() for l in layers]
layer_names = [l[0] for l in layers]

y_pos = range(len(layers))
widths = [c / max(layer_counts) for c in layer_counts]

for i, (name, width, count, color) in enumerate(zip(layer_names, widths, layer_counts, layer_colors)):
    ax.barh(i, width * 2, left=1 - width, height=0.7, 
            color=color, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.text(1, i, f'{name}: {count:,} ({count/len(user_profile)*100:.1f}%)', 
            ha='center', va='center', fontsize=10, fontweight='bold')

ax.set_xlim(0, 2)
ax.set_ylim(-0.5, len(layers) - 0.5)
ax.set_yticks([])
ax.set_xlabel('用户占比', fontsize=11, fontweight='bold')
ax.set_title('用户价值分层金字塔', fontsize=13, fontweight='bold', pad=12)
ax.grid(True, alpha=0.15, axis='x', linestyle='--')

ax = axes[1]
layer_features = ['活跃天数', '客单价', '留存天数', '整体转化率']
layer_data = []
for name, mask in layers:
    layer_users = user_profile[mask]
    features = [layer_users['活跃天数'].mean(), 
                layer_users['客单价'].mean(),
                layer_users['留存天数'].mean(),
                layer_users['整体转化率'].mean() * 100]
    layer_data.append(features)

layer_df = pd.DataFrame(layer_data, columns=layer_features, index=layer_names)
layer_df_norm = (layer_df - layer_df.min()) / (layer_df.max() - layer_df.min())

layer_df_norm.plot(kind='bar', ax=ax, color=layer_colors, 
                   alpha=0.85, edgecolor='white', linewidth=1.5)
ax.set_xlabel('用户分层', fontsize=11, fontweight='bold')
ax.set_ylabel('归一化值', fontsize=11, fontweight='bold')
ax.set_title('各层用户特征对比', fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.15, axis='y', linestyle='--')
ax.tick_params(axis='x', rotation=0)

fig.suptitle('用户分层与特征分析', fontsize=14, fontweight='bold', y=1.02)
fig.text(0.02, 0.02, '数据来源：电商平台用户行为数据 | 分层基于购买频次百分位数', 
         fontsize=8, style='italic', alpha=0.6, transform=fig.transFigure)
save_fig(fig, 'fig20_user_segmentation.png')

print("\n" + "="*60)
print("高级科研风格可视化完成！")
print("="*60)
print("\n生成的图表:")
for i in range(13, 21):
    print(f"  fig{i}_*.png")
print(f"\n输出目录: {OUTPUT_DIR}")
