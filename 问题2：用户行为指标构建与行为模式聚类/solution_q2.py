# -*- coding: utf-8 -*-
"""
问题2: 用户行为分析与行为模式聚类
- 构建用户行为指标体系
- 用户行为模式聚类分析
- 不同行为模式消费特征分析
- 科研风格可视化
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy import stats
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
DATA_DIR = r'g:\A.lsit\08.建模\校赛-B 题  销售商品行为预测\B题：附件1\赛题数据'
OUTPUT_DIR = r'g:\A.lsit\08.建模\校赛-B 题  销售商品行为预测\问题2'
Q1_DIR = r'g:\A.lsit\08.建模\校赛-B 题  销售商品行为预测\问题1'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置科研风格绘图
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
    'heatmap': 'RdYlBu_r',
    'diverging': 'coolwarm',
}

def save_fig(fig, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] 已保存: {filename}")

# ==================== 1. 加载数据 ====================
print("="*60)
print("步骤1: 加载数据")
print("="*60)

customers = pd.read_csv(os.path.join(DATA_DIR, 'customers_info.csv'))
behaviors = pd.read_csv(os.path.join(DATA_DIR, 'user_behavior.csv'))
products = pd.read_csv(os.path.join(DATA_DIR, 'products.csv'))
promotions = pd.read_csv(os.path.join(DATA_DIR, 'promotions.csv'))
locations = pd.read_csv(os.path.join(DATA_DIR, 'locations.csv'))

customers.columns = ['用户ID', '用户所在地', '年龄', '性别', '权益']
behaviors.columns = ['用户ID', '商品ID', '时间', '行为', '备注']
products.columns = ['商品ID', '商品类型', '具体商品', '成本', '单价', '库存']
promotions.columns = ['日期', '周末', '假期', '折扣量']
locations.columns = ['省份', '城市', '经济权重']

behaviors['时间'] = pd.to_datetime(behaviors['时间'], errors='coerce')
promotions['日期'] = pd.to_datetime(promotions['日期'], errors='coerce')

customers['性别'] = customers['性别'].fillna('未知')
customers['年龄'] = customers['年龄'].clip(lower=0, upper=120)
customers['年龄'] = customers['年龄'].fillna(customers['年龄'].median())
products['单价'] = products['单价'].fillna(products['单价'].median())

locations['城市'] = locations['城市'].fillna('未知')
locations['经济权重'] = locations['经济权重'].fillna(locations['经济权重'].median())

behaviors = behaviors.dropna(subset=['时间'])
behaviors = behaviors[behaviors['行为'].isin(['浏览', '收藏', '加购', '购买'])]

print(f"customers: {customers.shape}")
print(f"behaviors: {behaviors.shape}")
print(f"products: {products.shape}")

# ==================== 2. 数据合并 ====================
print("\n" + "="*60)
print("步骤2: 数据合并")
print("="*60)

df = behaviors.merge(customers, on='用户ID', how='left')
df = df.merge(products, on='商品ID', how='left')
df = df.merge(locations, left_on='用户所在地', right_on='城市', how='left')

df['月份'] = df['时间'].dt.month
df['星期'] = df['时间'].dt.dayofweek
df['小时'] = df['时间'].dt.hour
df['日期'] = df['时间'].dt.date
df['周末'] = df['时间'].dt.dayofweek >= 5

df['年龄段'] = pd.cut(df['年龄'], bins=[0, 20, 30, 40, 50, 60, 120],
                       labels=['<20', '20-30', '30-40', '40-50', '50-60', '60+'])

print(f"合并后数据集: {df.shape}")

# ==================== 3. 构建用户行为指标 ====================
print("\n" + "="*60)
print("步骤3: 构建用户行为指标")
print("="*60)

user_metrics = {}

# 3.1 基础行为指标
behavior_counts = df.groupby(['用户ID', '行为']).size().unstack(fill_value=0)
behavior_counts.columns = [f'{col}_次数' for col in behavior_counts.columns]
user_metrics['基础行为'] = behavior_counts

# 3.2 活跃度指标
user_dates = df.groupby('用户ID')['日期'].nunique().reset_index()
user_dates.columns = ['用户ID', '活跃天数']
user_metrics['活跃度'] = user_dates

# 3.3 消费指标
purchase_df = df[df['行为'] == '购买']
purchase_stats = purchase_df.groupby('用户ID').agg({
    '单价': ['count', 'sum', 'mean'],
    '商品ID': 'nunique',
    '商品类型': lambda x: x.mode()[0] if len(x.mode()) > 0 else '未知',
    '月份': ['min', 'max'],
}).reset_index()
purchase_stats.columns = ['用户ID', '购买次数', '总消费金额', '客单价', 
                          '购买商品种类数', '偏好商品类型', '首次购买月份', '最后购买月份']
user_metrics['消费指标'] = purchase_stats

# 3.4 留存指标
user_first_last = df.groupby('用户ID')['时间'].agg(['min', 'max']).reset_index()
user_first_last.columns = ['用户ID', '首次活跃时间', '最后活跃时间']
user_first_last['留存天数'] = (user_first_last['最后活跃时间'] - user_first_last['首次活跃时间']).dt.days
user_metrics['留存指标'] = user_first_last

# 3.5 跳失率（只浏览不购买的用户比例）
browse_only = df[df['行为'] == '浏览'].groupby('用户ID').size().reset_index(name='浏览次数')
purchase_users = df[df['行为'] == '购买']['用户ID'].unique()
browse_only['是否跳失'] = ~browse_only['用户ID'].isin(purchase_users)
user_metrics['跳失率'] = browse_only

# 3.6 行为转化率
behavior_order = ['浏览', '收藏', '加购', '购买']
user_behavior_seq = df.groupby(['用户ID', '行为']).size().unstack(fill_value=0)
for col in behavior_order:
    if col not in user_behavior_seq.columns:
        user_behavior_seq[col] = 0
user_behavior_seq = user_behavior_seq[behavior_order]
user_metrics['行为序列'] = user_behavior_seq

# 合并所有指标
user_profile = behavior_counts.reset_index()
user_profile = user_profile.merge(user_dates, on='用户ID', how='outer')
user_profile = user_profile.merge(purchase_stats, on='用户ID', how='outer')
user_profile = user_profile.merge(user_first_last[['用户ID', '留存天数']], on='用户ID', how='outer')
user_profile = user_profile.merge(customers[['用户ID', '年龄', '性别', '权益', '用户所在地']], on='用户ID', how='left')
user_profile = user_profile.merge(locations[['城市', '经济权重']], left_on='用户所在地', right_on='城市', how='left')

# 添加年龄段
user_profile['年龄段'] = pd.cut(user_profile['年龄'], bins=[0, 20, 30, 40, 50, 60, 120],
                                 labels=['<20', '20-30', '30-40', '40-50', '50-60', '60+'])

# 填充缺失值
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
user_profile['经济权重'] = user_profile['经济权重'].fillna(user_profile['经济权重'].median())

# 计算转化率
user_profile['浏览到收藏转化率'] = user_profile.apply(
    lambda x: x['收藏_次数'] / x['浏览_次数'] if x['浏览_次数'] > 0 else 0, axis=1)
user_profile['收藏到加购转化率'] = user_profile.apply(
    lambda x: x['加购_次数'] / x['收藏_次数'] if x['收藏_次数'] > 0 else 0, axis=1)
user_profile['加购到购买转化率'] = user_profile.apply(
    lambda x: x['购买_次数'] / x['加购_次数'] if x['加购_次数'] > 0 else 0, axis=1)
user_profile['整体转化率'] = user_profile.apply(
    lambda x: x['购买_次数'] / x['浏览_次数'] if x['浏览_次数'] > 0 else 0, axis=1)

# 计算跳失率
user_profile['是否跳失'] = user_profile['购买_次数'] == 0

print(f"用户画像数据集: {user_profile.shape}")
print(f"总用户数: {len(user_profile)}")
print(f"购买用户数: {(user_profile['购买_次数'] > 0).sum()}")
print(f"跳失用户数: {user_profile['是否跳失'].sum()}")
print(f"整体跳失率: {user_profile['是否跳失'].mean():.2%}")

# 保存用户画像数据
user_profile.to_csv(os.path.join(OUTPUT_DIR, 'user_profile.csv'), index=False, encoding='utf-8-sig')
print(f"\n用户画像数据已保存至: user_profile.csv")

# ==================== 4. 用户行为模式聚类 ====================
print("\n" + "="*60)
print("步骤4: 用户行为模式聚类分析")
print("="*60)

# 选择聚类特征
cluster_features = ['浏览_次数', '收藏_次数', '加购_次数', '购买_次数', 
                    '活跃天数', '总消费金额', '客单价', '留存天数',
                    '浏览到收藏转化率', '收藏到加购转化率', '加购到购买转化率']

# 过滤有效数据
cluster_df = user_profile[cluster_features].copy()
cluster_df = cluster_df.replace([np.inf, -np.inf], np.nan)
cluster_df = cluster_df.fillna(0)

# 标准化
scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_df)

# 确定最佳聚类数
print("\n--- 确定最佳聚类数 ---")
k_range = range(2, 11)
inertias = []
silhouette_scores = []
ch_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(cluster_scaled)
    inertias.append(kmeans.inertia_)
    sil = silhouette_score(cluster_scaled, labels, sample_size=min(5000, len(cluster_scaled)))
    silhouette_scores.append(sil)
    ch = calinski_harabasz_score(cluster_scaled, labels)
    ch_scores.append(ch)
    print(f"  K={k}: Inertia={kmeans.inertia_:.0f}, Silhouette={sil:.4f}, CH={ch:.0f}")

best_k = np.argmax(silhouette_scores) + 2
print(f"\n最佳聚类数: K={best_k} (轮廓系数={silhouette_scores[best_k-2]:.4f})")

# 执行最终聚类
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
user_profile['行为模式'] = kmeans_final.fit_predict(cluster_scaled)

# 分析各聚类特征
print("\n--- 各行为模式特征 ---")
cluster_summary = user_profile.groupby('行为模式')[cluster_features].mean()
print(cluster_summary.round(2))

# 保存聚类结果
user_profile.to_csv(os.path.join(OUTPUT_DIR, 'user_profile_with_clusters.csv'), index=False, encoding='utf-8-sig')

# ==================== 5. 可视化 ====================
print("\n" + "="*60)
print("步骤5: 生成可视化图表")
print("="*60)

# 图1: 用户行为指标分布箱线图
print("\n生成图1: 用户行为指标分布...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

metrics_to_plot = ['浏览_次数', '收藏_次数', '加购_次数', '购买_次数', '活跃天数', '留存天数']
metric_labels = ['浏览次数', '收藏次数', '加购次数', '购买次数', '活跃天数', '留存天数']

for i, (col, label) in enumerate(zip(metrics_to_plot, metric_labels)):
    ax = axes[i]
    data = user_profile[col].clip(upper=user_profile[col].quantile(0.95))
    bp = ax.boxplot(data, patch_artist=True, widths=0.5)
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['palette'][i % len(COLORS['palette'])])
        patch.set_alpha(0.7)
    ax.set_ylabel(label, fontsize=11, fontweight='bold')
    ax.set_xticks([1])
    ax.set_xticklabels(['全体用户'], fontsize=10)
    ax.grid(True, alpha=0.15, axis='y', linestyle='--')

axes[5].set_visible(False)
fig.suptitle('用户行为指标分布', fontsize=14, fontweight='bold', y=1.02)
fig.text(0.02, 0.02, '数据来源：电商平台用户行为数据 | 注：已去除5%极端值', 
         fontsize=8, style='italic', alpha=0.6, transform=fig.transFigure)
save_fig(fig, 'fig1_user_metrics_distribution.png')

# 图2: 行为转化率漏斗
print("生成图2: 行为转化率漏斗...")
fig, ax = plt.subplots(figsize=(10, 6))

behavior_order = ['浏览', '收藏', '加购', '购买']
behavior_counts_all = df.groupby('行为').size()
funnel_values = [behavior_counts_all.get(b, 0) for b in behavior_order]
funnel_labels = [f'{b}\n{v:,}' for b, v in zip(behavior_order, funnel_values)]

colors_funnel = [COLORS['palette'][0], COLORS['palette'][1], COLORS['palette'][2], COLORS['palette'][3]]
bars = ax.barh(range(len(funnel_values)), funnel_values, color=colors_funnel, 
               edgecolor='white', linewidth=1.5, height=0.6)

for i, (bar, val) in enumerate(zip(bars, funnel_values)):
    ax.text(val * 0.02, i, f'{val:,}', va='center', fontsize=10, fontweight='bold', color='white')
    if i > 0:
        rate = funnel_values[i] / funnel_values[i-1] * 100
        ax.text(val * 1.05, i, f'转化率: {rate:.1f}%', va='center', fontsize=9, 
                color=COLORS['accent'], fontweight='bold')

ax.set_yticks(range(len(funnel_values)))
ax.set_yticklabels(['浏览', '收藏', '加购', '购买'], fontsize=11, fontweight='bold')
ax.set_xlabel('行为次数', fontsize=12, fontweight='bold')
ax.set_title('用户行为转化漏斗', fontsize=14, fontweight='bold', pad=15)
ax.grid(True, alpha=0.15, axis='x', linestyle='--')
ax.set_xlim(0, max(funnel_values) * 1.3)
fig.text(0.02, 0.02, '数据来源：电商平台用户行为数据', 
         fontsize=8, style='italic', alpha=0.6, transform=fig.transFigure)
save_fig(fig, 'fig2_behavior_conversion_funnel.png')

# 图3: 聚类结果雷达图
print("生成图3: 行为模式雷达图...")
fig = plt.figure(figsize=(12, 10))

radar_features = ['浏览_次数', '收藏_次数', '加购_次数', '购买_次数', 
                  '活跃天数', '总消费金额', '客单价', '留存天数']
radar_labels = ['浏览次数', '收藏次数', '加购次数', '购买次数', 
                '活跃天数', '总消费金额', '客单价', '留存天数']

# 标准化到0-1
radar_data = user_profile.groupby('行为模式')[radar_features].mean()
radar_scaled = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min())

num_vars = len(radar_labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

ax = plt.subplot(111, projection='polar')

for cluster_id in range(best_k):
    values = radar_scaled.loc[cluster_id].tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, markersize=6, 
            label=f'模式{cluster_id}', color=COLORS['palette'][cluster_id % len(COLORS['palette'])])
    ax.fill(angles, values, alpha=0.15, color=COLORS['palette'][cluster_id % len(COLORS['palette'])])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=10)
ax.set_ylim(0, 1)
ax.set_title('用户行为模式雷达图', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9, fontsize=10)
ax.grid(True, alpha=0.2)
fig.text(0.02, 0.02, '数据来源：电商平台用户行为数据 | 数值已归一化至[0,1]', 
         fontsize=8, style='italic', alpha=0.6, transform=fig.transFigure)
save_fig(fig, 'fig3_cluster_radar.png')

# 图4: 聚类特征热力图
print("生成图4: 聚类特征热力图...")
fig, ax = plt.subplots(figsize=(12, 8))

cluster_heatmap = user_profile.groupby('行为模式')[cluster_features].mean()
cluster_heatmap_norm = (cluster_heatmap - cluster_heatmap.min()) / (cluster_heatmap.max() - cluster_heatmap.min())

sns.heatmap(cluster_heatmap_norm.T, ax=ax, cmap='RdYlBu_r', annot=True, fmt='.2f',
            cbar_kws={'label': '归一化值'}, linewidths=0.5, linecolor='white',
            xticklabels=[f'模式{i}' for i in range(best_k)])

ax.set_xlabel('行为模式', fontsize=12, fontweight='bold')
ax.set_ylabel('行为指标', fontsize=12, fontweight='bold')
ax.set_title('用户行为模式特征热力图', fontsize=14, fontweight='bold', pad=15)
fig.text(0.02, 0.02, '数据来源：电商平台用户行为数据 | 数值已归一化', 
         fontsize=8, style='italic', alpha=0.6, transform=fig.transFigure)
save_fig(fig, 'fig4_cluster_heatmap.png')

# 图5: 聚类结果PCA降维可视化
print("生成图5: PCA降维可视化...")
fig, ax = plt.subplots(figsize=(10, 8))

pca = PCA(n_components=2)
pca_result = pca.fit_transform(cluster_scaled)

scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], 
                     c=user_profile['行为模式'], cmap='tab10', 
                     alpha=0.6, s=30, edgecolors='white', linewidth=0.5)

ax.set_xlabel(f'主成分1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12, fontweight='bold')
ax.set_ylabel(f'主成分2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12, fontweight='bold')
ax.set_title('用户行为模式PCA降维可视化', fontsize=14, fontweight='bold', pad=15)

legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=COLORS['palette'][i % len(COLORS['palette'])], 
                               markersize=10, label=f'模式{i}') for i in range(best_k)]
ax.legend(handles=legend_elements, loc='best', framealpha=0.9, fontsize=10)
ax.grid(True, alpha=0.15, linestyle='--')
fig.text(0.02, 0.02, '数据来源：电商平台用户行为数据', 
         fontsize=8, style='italic', alpha=0.6, transform=fig.transFigure)
save_fig(fig, 'fig5_pca_visualization.png')

# 图6: 不同行为模式消费特征对比
print("生成图6: 行为模式消费特征对比...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 6a: 购买次数分布
ax = axes[0, 0]
cluster_purchase = user_profile.groupby('行为模式')['购买_次数'].agg(['mean', 'median', 'std']).reset_index()
x = np.arange(len(cluster_purchase))
width = 0.25

bars1 = ax.bar(x - width, cluster_purchase['mean'], width, 
               color=COLORS['palette'][:best_k], alpha=0.85, label='均值',
               edgecolor='white', linewidth=1.5)
bars2 = ax.bar(x, cluster_purchase['median'], width, 
               color=[c + '99' for c in COLORS['palette'][:best_k]], alpha=0.85, label='中位数',
               edgecolor='white', linewidth=1.5)

ax.set_xticks(x)
ax.set_xticklabels([f'模式{i}' for i in range(best_k)], fontsize=11)
ax.set_ylabel('购买次数', fontsize=11, fontweight='bold')
ax.set_title('各模式购买次数对比', fontsize=12, fontweight='bold', pad=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.15, axis='y', linestyle='--')

# 6b: 客单价分布
ax = axes[0, 1]
cluster_aov = user_profile.groupby('行为模式')['客单价'].agg(['mean', 'median']).reset_index()
bars = ax.bar(cluster_aov['行为模式'], cluster_aov['mean'], 
              color=COLORS['palette'][:best_k], alpha=0.85,
              edgecolor='white', linewidth=1.5)

for bar, val in zip(bars, cluster_aov['mean']):
    ax.annotate(f'{val:.0f}元',
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 8), textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(range(best_k))
ax.set_xticklabels([f'模式{i}' for i in range(best_k)], fontsize=11)
ax.set_ylabel('客单价 (元)', fontsize=11, fontweight='bold')
ax.set_title('各模式客单价对比', fontsize=12, fontweight='bold', pad=12)
ax.grid(True, alpha=0.15, axis='y', linestyle='--')

# 6c: 总消费金额
ax = axes[1, 0]
cluster_total = user_profile.groupby('行为模式')['总消费金额'].sum().reset_index()
cluster_total.columns = ['行为模式', '总消费金额']
bars = ax.bar(cluster_total['行为模式'], cluster_total['总消费金额'] / 1000, 
              color=COLORS['palette'][:best_k], alpha=0.85,
              edgecolor='white', linewidth=1.5)

for bar, val in zip(bars, cluster_total['总消费金额']):
    ax.annotate(f'{val/1000:.0f}K',
                xy=(bar.get_x() + bar.get_width() / 2, val/1000),
                xytext=(0, 8), textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(range(best_k))
ax.set_xticklabels([f'模式{i}' for i in range(best_k)], fontsize=11)
ax.set_ylabel('总消费金额 (千元)', fontsize=11, fontweight='bold')
ax.set_title('各模式总消费金额对比', fontsize=12, fontweight='bold', pad=12)
ax.grid(True, alpha=0.15, axis='y', linestyle='--')

# 6d: 转化率对比
ax = axes[1, 1]
cluster_conversion = user_profile.groupby('行为模式')['整体转化率'].mean().reset_index()
bars = ax.bar(cluster_conversion['行为模式'], cluster_conversion['整体转化率'] * 100, 
              color=COLORS['palette'][:best_k], alpha=0.85,
              edgecolor='white', linewidth=1.5)

for bar, val in zip(bars, cluster_conversion['整体转化率']):
    ax.annotate(f'{val*100:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, val*100),
                xytext=(0, 8), textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(range(best_k))
ax.set_xticklabels([f'模式{i}' for i in range(best_k)], fontsize=11)
ax.set_ylabel('整体转化率 (%)', fontsize=11, fontweight='bold')
ax.set_title('各模式转化率对比', fontsize=12, fontweight='bold', pad=12)
ax.grid(True, alpha=0.15, axis='y', linestyle='--')

fig.suptitle('不同行为模式消费特征对比', fontsize=14, fontweight='bold', y=1.02)
fig.text(0.02, 0.02, '数据来源：电商平台用户行为数据', 
         fontsize=8, style='italic', alpha=0.6, transform=fig.transFigure)
save_fig(fig, 'fig6_cluster_consumption.png')

# 图7: 行为模式与用户属性关系
print("生成图7: 行为模式与用户属性关系...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 7a: 年龄段分布
ax = axes[0]
age_cluster = pd.crosstab(user_profile['年龄段'], user_profile['行为模式'], normalize='index')
age_cluster.plot(kind='bar', ax=ax, color=COLORS['palette'][:best_k], 
                 alpha=0.85, edgecolor='white', linewidth=1.5)
ax.set_xlabel('年龄段', fontsize=11, fontweight='bold')
ax.set_ylabel('占比', fontsize=11, fontweight='bold')
ax.set_title('年龄段与行为模式分布', fontsize=12, fontweight='bold', pad=12)
ax.legend(title='行为模式', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.15, axis='y', linestyle='--')

# 7b: 性别分布
ax = axes[1]
gender_cluster = pd.crosstab(user_profile['性别'], user_profile['行为模式'], normalize='index')
gender_cluster.plot(kind='bar', ax=ax, color=COLORS['palette'][:best_k], 
                    alpha=0.85, edgecolor='white', linewidth=1.5)
ax.set_xlabel('性别', fontsize=11, fontweight='bold')
ax.set_ylabel('占比', fontsize=11, fontweight='bold')
ax.set_title('性别与行为模式分布', fontsize=12, fontweight='bold', pad=12)
ax.legend(title='行为模式', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.15, axis='y', linestyle='--')

# 7c: 权益分布
ax = axes[2]
membership_cluster = pd.crosstab(user_profile['权益'], user_profile['行为模式'], normalize='index')
membership_cluster.plot(kind='bar', ax=ax, color=COLORS['palette'][:best_k], 
                        alpha=0.85, edgecolor='white', linewidth=1.5)
ax.set_xlabel('权益类型', fontsize=11, fontweight='bold')
ax.set_ylabel('占比', fontsize=11, fontweight='bold')
ax.set_title('权益与行为模式分布', fontsize=12, fontweight='bold', pad=12)
ax.legend(title='行为模式', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.15, axis='y', linestyle='--')

fig.suptitle('行为模式与用户属性关系', fontsize=14, fontweight='bold', y=1.02)
fig.text(0.02, 0.02, '数据来源：电商平台用户行为数据', 
         fontsize=8, style='italic', alpha=0.6, transform=fig.transFigure)
save_fig(fig, 'fig7_cluster_demographics.png')

# 图8: 聚类质量评估
print("生成图8: 聚类质量评估...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 8a: 肘部法则
ax = axes[0]
ax.plot(k_range, inertias, 'o-', linewidth=2, markersize=8, color=COLORS['primary'])
ax.axvline(x=best_k, color=COLORS['accent'], linestyle='--', alpha=0.7, label=f'最佳K={best_k}')
ax.set_xlabel('聚类数 K', fontsize=12, fontweight='bold')
ax.set_ylabel('惯性 (Inertia)', fontsize=12, fontweight='bold')
ax.set_title('肘部法则', fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.15, linestyle='--')

# 8b: 轮廓系数
ax = axes[1]
ax.plot(k_range, silhouette_scores, 's-', linewidth=2, markersize=8, color=COLORS['secondary'])
ax.axvline(x=best_k, color=COLORS['accent'], linestyle='--', alpha=0.7, label=f'最佳K={best_k}')
ax.set_xlabel('聚类数 K', fontsize=12, fontweight='bold')
ax.set_ylabel('轮廓系数', fontsize=12, fontweight='bold')
ax.set_title('轮廓系数分析', fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.15, linestyle='--')

fig.suptitle('聚类质量评估', fontsize=14, fontweight='bold', y=1.02)
fig.text(0.02, 0.02, '数据来源：电商平台用户行为数据', 
         fontsize=8, style='italic', alpha=0.6, transform=fig.transFigure)
save_fig(fig, 'fig8_cluster_evaluation.png')

# 图9: 行为模式偏好商品类型
print("生成图9: 行为模式偏好商品类型...")
fig, ax = plt.subplots(figsize=(12, 8))

purchase_with_cluster = purchase_df.merge(user_profile[['用户ID', '行为模式']], on='用户ID', how='left')
cluster_category = pd.crosstab(purchase_with_cluster['行为模式'], purchase_with_cluster['商品类型'], normalize='index')

cluster_category.plot(kind='bar', ax=ax, color=COLORS['palette'][:best_k], 
                      alpha=0.85, edgecolor='white', linewidth=1.5)
ax.set_xlabel('行为模式', fontsize=12, fontweight='bold')
ax.set_ylabel('占比', fontsize=12, fontweight='bold')
ax.set_title('各行为模式偏好商品类型分布', fontsize=14, fontweight='bold', pad=15)
ax.legend(title='商品类型', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.15, axis='y', linestyle='--')
ax.tick_params(axis='x', rotation=0)
fig.text(0.02, 0.02, '数据来源：电商平台用户行为数据', 
         fontsize=8, style='italic', alpha=0.6, transform=fig.transFigure)
save_fig(fig, 'fig9_cluster_category_preference.png')

# 图10: 用户活跃度与消费关系散点图
print("生成图10: 用户活跃度与消费关系...")
fig, ax = plt.subplots(figsize=(10, 8))

# 采样以避免过度绘制
sample_size = min(5000, len(user_profile))
sample_idx = np.random.choice(len(user_profile), sample_size, replace=False)
sample_df = user_profile.iloc[sample_idx]

scatter = ax.scatter(sample_df['活跃天数'], sample_df['总消费金额'] / 1000, 
                     c=sample_df['行为模式'], cmap='tab10', 
                     alpha=0.5, s=40, edgecolors='white', linewidth=0.5)

ax.set_xlabel('活跃天数', fontsize=12, fontweight='bold')
ax.set_ylabel('总消费金额 (千元)', fontsize=12, fontweight='bold')
ax.set_title('用户活跃度与消费金额关系', fontsize=14, fontweight='bold', pad=15)

legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=COLORS['palette'][i % len(COLORS['palette'])], 
                               markersize=10, label=f'模式{i}') for i in range(best_k)]
ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=10)
ax.grid(True, alpha=0.15, linestyle='--')
fig.text(0.02, 0.02, f'数据来源：电商平台用户行为数据 | 采样{sample_size}个用户', 
         fontsize=8, style='italic', alpha=0.6, transform=fig.transFigure)
save_fig(fig, 'fig10_activity_vs_consumption.png')

# 图11: 行为模式用户留存分析
print("生成图11: 行为模式用户留存分析...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 11a: 留存天数分布
ax = axes[0]
for cluster_id in range(best_k):
    cluster_data = user_profile[user_profile['行为模式'] == cluster_id]['留存天数']
    ax.hist(cluster_data, bins=50, alpha=0.6, label=f'模式{cluster_id}',
            color=COLORS['palette'][cluster_id % len(COLORS['palette'])],
            edgecolor='white', linewidth=0.5)

ax.set_xlabel('留存天数', fontsize=12, fontweight='bold')
ax.set_ylabel('用户数', fontsize=12, fontweight='bold')
ax.set_title('各行为模式留存天数分布', fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.15, axis='y', linestyle='--')

# 11b: 平均留存天数对比
ax = axes[1]
cluster_retention = user_profile.groupby('行为模式')['留存天数'].agg(['mean', 'median']).reset_index()
x = np.arange(len(cluster_retention))
width = 0.35

bars1 = ax.bar(x - width/2, cluster_retention['mean'], width, 
               color=COLORS['palette'][:best_k], alpha=0.85, label='均值',
               edgecolor='white', linewidth=1.5)
bars2 = ax.bar(x + width/2, cluster_retention['median'], width, 
               color=[c + '99' for c in COLORS['palette'][:best_k]], alpha=0.85, label='中位数',
               edgecolor='white', linewidth=1.5)

ax.set_xticks(x)
ax.set_xticklabels([f'模式{i}' for i in range(best_k)], fontsize=11)
ax.set_ylabel('留存天数', fontsize=11, fontweight='bold')
ax.set_title('各模式平均留存天数对比', fontsize=12, fontweight='bold', pad=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.15, axis='y', linestyle='--')

fig.suptitle('用户留存分析', fontsize=14, fontweight='bold', y=1.02)
fig.text(0.02, 0.02, '数据来源：电商平台用户行为数据', 
         fontsize=8, style='italic', alpha=0.6, transform=fig.transFigure)
save_fig(fig, 'fig11_retention_analysis.png')

# 图12: 行为模式综合画像
print("生成图12: 行为模式综合画像...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 12a: 各模式用户占比
ax = axes[0, 0]
cluster_counts = user_profile['行为模式'].value_counts().sort_index()
wedges, texts, autotexts = ax.pie(cluster_counts.values, 
                                    labels=[f'模式{i}' for i in cluster_counts.index],
                                    autopct='%1.1f%%',
                                    colors=COLORS['palette'][:best_k],
                                    startangle=90,
                                    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2))
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(9)
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
ax.add_artist(centre_circle)
ax.set_title('各行为模式用户占比', fontsize=12, fontweight='bold', pad=12)

# 12b: 跳失率对比
ax = axes[0, 1]
cluster_bounce = user_profile.groupby('行为模式')['是否跳失'].mean() * 100
bars = ax.bar(range(len(cluster_bounce)), cluster_bounce.values, 
              color=COLORS['palette'][:best_k], alpha=0.85,
              edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, cluster_bounce.values):
    ax.annotate(f'{val:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 8), textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(range(len(cluster_bounce)))
ax.set_xticklabels([f'模式{i}' for i in range(len(cluster_bounce))], fontsize=11)
ax.set_ylabel('跳失率 (%)', fontsize=11, fontweight='bold')
ax.set_title('各模式跳失率对比', fontsize=12, fontweight='bold', pad=12)
ax.grid(True, alpha=0.15, axis='y', linestyle='--')
ax.set_ylim(0, 100)

# 12c: 购买用户占比
ax = axes[1, 0]
cluster_purchase_rate = user_profile.groupby('行为模式').apply(
    lambda x: (x['购买_次数'] > 0).sum() / len(x) * 100)
bars = ax.bar(range(len(cluster_purchase_rate)), cluster_purchase_rate.values, 
              color=COLORS['palette'][:best_k], alpha=0.85,
              edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, cluster_purchase_rate.values):
    ax.annotate(f'{val:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 8), textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(range(len(cluster_purchase_rate)))
ax.set_xticklabels([f'模式{i}' for i in range(len(cluster_purchase_rate))], fontsize=11)
ax.set_ylabel('购买用户占比 (%)', fontsize=11, fontweight='bold')
ax.set_title('各模式购买用户占比', fontsize=12, fontweight='bold', pad=12)
ax.grid(True, alpha=0.15, axis='y', linestyle='--')
ax.set_ylim(0, 100)

# 12d: 平均活跃天数
ax = axes[1, 1]
cluster_activity = user_profile.groupby('行为模式')['活跃天数'].mean()
bars = ax.bar(range(len(cluster_activity)), cluster_activity.values, 
              color=COLORS['palette'][:best_k], alpha=0.85,
              edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, cluster_activity.values):
    ax.annotate(f'{val:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 8), textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(range(len(cluster_activity)))
ax.set_xticklabels([f'模式{i}' for i in range(len(cluster_activity))], fontsize=11)
ax.set_ylabel('平均活跃天数', fontsize=11, fontweight='bold')
ax.set_title('各模式平均活跃天数', fontsize=12, fontweight='bold', pad=12)
ax.grid(True, alpha=0.15, axis='y', linestyle='--')

fig.suptitle('用户行为模式综合画像', fontsize=14, fontweight='bold', y=1.02)
fig.text(0.02, 0.02, '数据来源：电商平台用户行为数据', 
         fontsize=8, style='italic', alpha=0.6, transform=fig.transFigure)
save_fig(fig, 'fig12_cluster_profile.png')

print("\n" + "="*60)
print("问题2分析完成！")
print("="*60)
print(f"\n生成的图表:")
for i in range(1, 13):
    print(f"  fig{i}_*.png")
print(f"\n输出目录: {OUTPUT_DIR}")
