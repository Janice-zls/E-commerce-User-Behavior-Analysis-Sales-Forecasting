# -*- coding: utf-8 -*-
"""
问题1: 高级科研风格可视化 - 高效版
使用采样加速大数据集处理
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import seaborn as sns
from scipy import stats
import warnings
import os
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
DATA_DIR = r'g:\A.lsit\08.建模\校赛-B 题  销售商品行为预测\B题：附件1\赛题数据'
OUTPUT_DIR = r'g:\A.lsit\08.建模\校赛-B 题  销售商品行为预测\问题1'

# 设置中文字体
plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei'],
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

# 配色方案
COLORS = {
    'primary': '#1B4F72',
    'secondary': '#2E86C1',
    'accent': '#C0392B',
    'success': '#27AE60',
    'warning': '#F39C12',
    'palette': ['#1B4F72', '#2E86C1', '#C0392B', '#27AE60', '#F39C12', '#8E44AD', '#16A085', '#D35400', 
                '#2C3E50', '#E74C3C', '#3498DB', '#2980B9'],
}

def save_fig(fig, filename):
    try:
        filepath = os.path.join(OUTPUT_DIR, filename)
        fig.tight_layout()
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"✓ 已保存: {filename}")
        return True
    except Exception as e:
        print(f"✗ 保存失败 {filename}: {e}")
        return False

# ==================== 数据加载 ====================
print("="*60)
print("加载数据")
print("="*60)

df = pd.read_csv(os.path.join(OUTPUT_DIR, 'cleaned_data.csv'), encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df['日期'] = pd.to_datetime(df['日期'])

purchase_df = df[df['行为'] == '购买'].copy()
print(f"总数据: {len(df)} 条, 购买数据: {len(purchase_df)} 条")

# 采样用于密度图等计算密集型可视化
np.random.seed(42)
if len(purchase_df) > 100000:
    purchase_sample = purchase_df.sample(n=100000, random_state=42)
    print(f"采样: 使用 {len(purchase_sample)} 条购买数据用于密度图")
else:
    purchase_sample = purchase_df.copy()

# ==================== 图13: 用户行为与属性分析 ====================
print("\n生成图13: 用户行为与属性分析")

try:
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 3a: 行为转化漏斗
    ax1 = fig.add_subplot(gs[0, 0])
    behavior_counts = df.groupby('行为').size().reindex(['浏览', '收藏', '加购', '购买'], fill_value=0)
    max_val = behavior_counts.max()
    colors_funnel = [COLORS['palette'][i] for i in range(4)]

    for i, (behavior, count) in enumerate(behavior_counts.items()):
        width = count / max_val
        left = (1 - width) / 2
        ax1.barh(i, width, left=left, height=0.65, color=colors_funnel[i],
                 edgecolor='white', linewidth=2.5, alpha=0.9)
        ax1.text(0.5, i, f'{behavior}: {count:,}', ha='center', va='center',
                 fontsize=11, fontweight='bold', color='white')
        if i > 0:
            prev_count = behavior_counts.iloc[i-1]
            rate = count / prev_count * 100
            ax1.text(0.97, i - 0.3, f'转化率: {rate:.1f}%',
                     ha='right', va='center', fontsize=9,
                     style='italic', color=COLORS['accent'], fontweight='bold')

    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.5, 3.5)
    ax1.set_yticks(range(4))
    ax1.set_yticklabels([])
    ax1.axis('off')
    ax1.set_title('用户行为转化漏斗', fontsize=13, fontweight='bold', pad=10)

    # 3b: 年龄-购买关系（2D密度图）- 使用采样
    ax2 = fig.add_subplot(gs[0, 1])
    x_data = purchase_sample['年龄']
    y_data = purchase_sample['单价']

    sns.kdeplot(x=x_data, y=y_data, ax=ax2, cmap='YlOrRd', 
                fill=True, levels=20, alpha=0.8)
    ax2.set_xlabel('年龄', fontsize=12, fontweight='bold')
    ax2.set_ylabel('客单价 (元)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.15, linestyle='--')
    ax2.set_title('年龄-客单价密度分布', fontsize=13, fontweight='bold', pad=10)

    # 3c: 性别×权益×购买
    ax3 = fig.add_subplot(gs[0, 2])
    gender_membership_data = []
    labels = []
    for gender in ['男', '女']:
        for membership in ['普通', '会员']:
            subset = purchase_df[(purchase_df['性别'] == gender) & (purchase_df['权益'] == membership)]['单价']
            if len(subset) > 0:
                gender_membership_data.append(subset.values)
                labels.append(f'{gender}-{membership}')

    parts = ax3.violinplot(gender_membership_data, positions=range(len(labels)), widths=0.6,
                           showmeans=True, showmedians=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(COLORS['palette'][i % len(COLORS['palette'])])
        pc.set_alpha(0.6)
        pc.set_edgecolor('white')

    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, fontsize=10)
    ax3.set_ylabel('客单价 (元)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.15, axis='y', linestyle='--')
    ax3.set_title('性别-权益客单价分布', fontsize=13, fontweight='bold', pad=10)

    # 3d: 支付方式分布
    ax4 = fig.add_subplot(gs[1, 0])
    payment_data = purchase_df[purchase_df['备注'].notna() & (purchase_df['备注'] != '')]
    payment_counts = payment_data['备注'].value_counts()

    wedges, texts, autotexts = ax4.pie(payment_counts.values,
                                        labels=payment_counts.index,
                                        autopct='%1.1f%%',
                                        colors=COLORS['palette'][:len(payment_counts)],
                                        startangle=90,
                                        pctdistance=0.85,
                                        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2))

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax4.add_artist(centre_circle)
    ax4.set_title('支付方式分布', fontsize=13, fontweight='bold', pad=10)

    # 3e: 会员vs普通用户
    ax5 = fig.add_subplot(gs[1, 1])
    member_data = purchase_df[purchase_df['权益'] == '会员']['单价']
    regular_data = purchase_df[purchase_df['权益'] == '普通']['单价']

    bp = ax5.boxplot([member_data.values, regular_data.values], 
                     labels=['会员', '普通'], patch_artist=True,
                     boxprops=dict(facecolor=COLORS['secondary'], alpha=0.7, edgecolor='white', linewidth=2),
                     medianprops=dict(color=COLORS['accent'], linewidth=2.5))

    ax5.set_ylabel('客单价 (元)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.15, axis='y', linestyle='--')
    ax5.set_title('会员vs普通用户客单价对比', fontsize=13, fontweight='bold', pad=10)

    # 3f: 年龄分布
    ax6 = fig.add_subplot(gs[1, 2])
    sns.histplot(data=purchase_sample, x='年龄', kde=True, ax=ax6,
                 color=COLORS['primary'], alpha=0.6, edgecolor='white', linewidth=1.5,
                 line_kws={'color': COLORS['accent'], 'linewidth': 2.5})
    ax6.set_xlabel('年龄', fontsize=12, fontweight='bold')
    ax6.set_ylabel('频数', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.15, axis='y', linestyle='--')
    ax6.set_title('购买用户年龄分布', fontsize=13, fontweight='bold', pad=10)

    fig.suptitle('用户行为与属性深度分析', fontsize=16, fontweight='bold', y=0.98)
    fig.text(0.02, 0.01, '数据来源：电商平台用户行为数据', fontsize=9, style='italic', alpha=0.6)
    save_fig(fig, 'fig13_user_behavior_analysis.png')
except Exception as e:
    print(f"图13生成失败: {e}")
    import traceback
    traceback.print_exc()

# ==================== 图14: 促销与地区影响分析 ====================
print("\n生成图14: 促销与地区影响分析")

try:
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # 4a: 折扣量与购买关系
    ax1 = fig.add_subplot(gs[0, 0])
    promotions_df = pd.read_csv(os.path.join(DATA_DIR, 'promotions.csv'), encoding='utf-8-sig')
    promotions_df.columns = ['日期', '周末', '假期', '折扣量']
    promotions_df['日期'] = pd.to_datetime(promotions_df['日期'])
    
    promo_purchase = purchase_df.merge(promotions_df, left_on='时间', right_on='日期', how='inner', suffixes=('', '_p'))

    discount_bins = pd.cut(promo_purchase['折扣量'], bins=[0, 0.8, 0.85, 0.9, 0.95, 1.0],
                           labels=['0-0.8', '0.8-0.85', '0.85-0.9', '0.9-0.95', '0.95-1.0'])
    discount_data = [group['单价'].values for name, group in promo_purchase.groupby(discount_bins)]

    parts = ax1.violinplot(discount_data, positions=range(5), widths=0.6,
                           showmeans=True, showmedians=True)

    for pc in parts['bodies']:
        pc.set_facecolor(COLORS['secondary'])
        pc.set_alpha(0.6)
        pc.set_edgecolor('white')

    ax1.set_xticks(range(5))
    ax1.set_xticklabels(['0-0.8', '0.8-0.85', '0.85-0.9', '0.9-0.95', '0.95-1.0'], fontsize=10)
    ax1.set_xlabel('折扣区间', fontsize=12, fontweight='bold')
    ax1.set_ylabel('客单价 (元)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.15, axis='y', linestyle='--')
    ax1.set_title('折扣区间对客单价的影响', fontsize=13, fontweight='bold', pad=10)

    # 4b: 周末vs工作日对比
    ax2 = fig.add_subplot(gs[0, 1])
    weekend_data = promo_purchase.groupby('周末').agg(
        购买次数=('用户ID', 'count'),
        平均客单价=('单价', 'mean')
    ).reset_index()
    weekend_data['类型'] = weekend_data['周末'].map({True: '周末', False: '工作日'})

    x = np.arange(2)
    width = 0.35

    bars1 = ax2.bar(x - width/2, weekend_data['购买次数'], width,
                    color=COLORS['primary'], alpha=0.85, label='购买次数',
                    edgecolor='white', linewidth=2)
    ax2.set_ylabel('购买次数', fontsize=12, fontweight='bold', color=COLORS['primary'])
    ax2.tick_params(axis='y', labelcolor=COLORS['primary'])

    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x + width/2, weekend_data['平均客单价'], width,
                         color=COLORS['accent'], alpha=0.85, label='平均客单价',
                         edgecolor='white', linewidth=2)
    ax2_twin.set_ylabel('平均客单价 (元)', fontsize=12, fontweight='bold', color=COLORS['accent'])
    ax2_twin.tick_params(axis='y', labelcolor=COLORS['accent'])

    ax2.set_xticks(x)
    ax2.set_xticklabels(['工作日', '周末'], fontsize=11)
    ax2.grid(True, alpha=0.15, axis='y', linestyle='--')

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.9)
    ax2.set_title('周末vs工作日销售对比', fontsize=13, fontweight='bold', pad=10)

    # 4c: 地区经济权重分布
    ax3 = fig.add_subplot(gs[1, 0])
    region_purchase = purchase_df.groupby('经济权重').agg(
        购买次数=('用户ID', 'count'),
        购买用户数=('用户ID', 'nunique'),
        平均客单价=('单价', 'mean')
    ).reset_index()

    scatter = ax3.scatter(region_purchase['经济权重'], region_purchase['购买次数'],
                          s=region_purchase['购买用户数']*3, alpha=0.7,
                          c=region_purchase['平均客单价'], cmap='RdBu_r',
                          edgecolors='white', linewidth=2, zorder=5)

    for i, txt in enumerate(region_purchase['经济权重']):
        ax3.annotate(f'{txt:.0f}', 
                     (region_purchase['经济权重'].iloc[i], region_purchase['购买次数'].iloc[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')

    ax3.set_xlabel('经济权重', fontsize=12, fontweight='bold')
    ax3.set_ylabel('购买次数', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.2, linestyle='--')
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('平均客单价 (元)', fontsize=10, fontweight='bold')
    ax3.set_title('地区经济权重与购买行为', fontsize=13, fontweight='bold', pad=10)

    # 4d: 假期效应分析
    ax4 = fig.add_subplot(gs[1, 1])
    calendar_data = promo_purchase.groupby(promo_purchase['时间'].dt.date).size()
    calendar_df = pd.DataFrame({'日期': calendar_data.index, '购买次数': calendar_data.values})
    calendar_df['月份'] = calendar_df['日期'].apply(lambda x: x.month)
    calendar_df['日'] = calendar_df['日期'].apply(lambda x: x.day)

    pivot_calendar = calendar_df.pivot(index='月份', columns='日', values='购买次数')
    sns.heatmap(pivot_calendar, ax=ax4, cmap='YlOrRd', 
                cbar_kws={'label': '购买次数'}, linewidths=0.3, linecolor='white',
                annot=False)
    ax4.set_xlabel('日期', fontsize=12, fontweight='bold')
    ax4.set_ylabel('月份', fontsize=12, fontweight='bold')
    ax4.set_title('日历热力图：每日购买分布', fontsize=13, fontweight='bold', pad=10)

    fig.suptitle('促销与地区影响深度分析', fontsize=16, fontweight='bold', y=0.98)
    fig.text(0.02, 0.01, '数据来源：电商平台用户行为数据', fontsize=9, style='italic', alpha=0.6)
    save_fig(fig, 'fig14_promotion_region_analysis.png')
except Exception as e:
    print(f"图14生成失败: {e}")
    import traceback
    traceback.print_exc()

# ==================== 图15: 统计检验结果可视化 ====================
print("\n生成图15: 统计检验结果可视化")

try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 5a: 商品类型ANOVA结果
    ax = axes[0, 0]
    category_groups = [group['单价'].values for name, group in purchase_df.groupby('商品类型')]
    f_stat, p_val = stats.f_oneway(*category_groups)

    category_means = purchase_df.groupby('商品类型')['单价'].mean().sort_values(ascending=False)
    colors_anova = [COLORS['palette'][i % len(COLORS['palette'])] for i in range(len(category_means))]

    bars = ax.barh(range(len(category_means)), category_means.values, 
                   color=colors_anova, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.set_yticks(range(len(category_means)))
    ax.set_yticklabels(category_means.index, fontsize=10)
    ax.set_xlabel('平均客单价 (元)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.15, axis='x', linestyle='--')

    ax.text(0.98, 0.98, f'ANOVA检验\nF={f_stat:.2f}\np<{0.001 if p_val < 0.001 else p_val:.4f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor=COLORS['accent'], alpha=0.2, edgecolor=COLORS['accent']))

    ax.set_title('商品类型客单价差异检验', fontsize=13, fontweight='bold', pad=10)

    # 5b: 会员vs普通用户t检验
    ax = axes[0, 1]
    member_prices = purchase_df[purchase_df['权益'] == '会员']['单价']
    regular_prices = purchase_df[purchase_df['权益'] == '普通']['单价']
    t_stat, t_pval = stats.ttest_ind(member_prices, regular_prices)

    data_to_plot = [member_prices.values, regular_prices.values]
    bp = ax.boxplot(data_to_plot, labels=['会员', '普通'], patch_artist=True,
                    boxprops=dict(facecolor=COLORS['secondary'], alpha=0.7, edgecolor='white', linewidth=2),
                    medianprops=dict(color=COLORS['accent'], linewidth=2.5))

    ax.set_ylabel('客单价 (元)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.15, axis='y', linestyle='--')

    ax.text(0.98, 0.98, f't检验\nt={t_stat:.2f}\np<{0.001 if t_pval < 0.001 else t_pval:.4f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor=COLORS['accent'], alpha=0.2, edgecolor=COLORS['accent']))

    ax.set_title('会员vs普通用户客单价检验', fontsize=13, fontweight='bold', pad=10)

    # 5c: 年龄与购买相关性
    ax = axes[1, 0]
    user_age_purchase = purchase_df.groupby('用户ID').agg(
        年龄=('年龄', 'first'),
        购买次数=('用户ID', 'count')
    ).reset_index()

    rho, rho_pval = stats.spearmanr(user_age_purchase['年龄'], user_age_purchase['购买次数'])

    sns.regplot(x='年龄', y='购买次数', data=user_age_purchase, ax=ax,
                scatter_kws={'alpha': 0.3, 's': 30, 'color': COLORS['secondary']},
                line_kws={'color': COLORS['accent'], 'linewidth': 2.5})

    ax.set_xlabel('年龄', fontsize=12, fontweight='bold')
    ax.set_ylabel('购买次数', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.15, linestyle='--')

    ax.text(0.02, 0.98, f'Spearman相关\nρ={rho:.3f}\np<{0.001 if rho_pval < 0.001 else rho_pval:.4f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=COLORS['accent'], alpha=0.2, edgecolor=COLORS['accent']))

    ax.set_title('年龄与购买次数相关性', fontsize=13, fontweight='bold', pad=10)

    # 5d: 促销影响
    ax = axes[1, 1]
    high_discount = promo_purchase[promo_purchase['折扣量'] < 0.9].shape[0]
    low_discount = promo_purchase[promo_purchase['折扣量'] >= 0.9].shape[0]

    bars = ax.bar(['高折扣(<0.9)', '低折扣(≥0.9)'], [high_discount, low_discount],
                  color=[COLORS['success'], COLORS['warning']], alpha=0.85,
                  edgecolor='white', linewidth=2)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height):,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 8), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('购买次数', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.15, axis='y', linestyle='--')
    ax.set_title('促销折扣影响分析', fontsize=13, fontweight='bold', pad=10)

    fig.suptitle('关键影响因素统计检验结果', fontsize=16, fontweight='bold', y=0.98)
    fig.text(0.02, 0.01, '数据来源：电商平台用户行为数据 | 显著性水平α=0.05', fontsize=9, style='italic', alpha=0.6)
    save_fig(fig, 'fig15_statistical_tests.png')
except Exception as e:
    print(f"图15生成失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("所有高级可视化图表生成完成!")
print("="*60)
