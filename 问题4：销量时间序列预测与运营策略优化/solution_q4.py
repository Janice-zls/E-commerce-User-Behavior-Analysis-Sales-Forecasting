# -*- coding: utf-8 -*-
"""
问题4：商品销量时间序列预测与营销策略优化
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy import stats
import warnings
import os
from collections import defaultdict

warnings.filterwarnings('ignore')

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

COLORS = {
    'primary': '#2C3E50',
    'secondary': '#3498DB',
    'accent': '#E74C3C',
    'success': '#27AE60',
    'warning': '#F39C12',
    'info': '#1ABC9C',
    'palette': ['#2C3E50', '#3498DB', '#E74C3C', '#27AE60', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22'],
}

OUTPUT_DIR = r'g:\A.lsit\08.建模\校赛-B 题  销售商品行为预测\问题4'
DATA_DIR = r'g:\A.lsit\08.建模\校赛-B 题  销售商品行为预测\B题：附件1\赛题数据'

def save_fig(fig, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig)
    print(f"  已保存: {filename}")

def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

print("=" * 60)
print("问题4：商品销量时间序列预测与营销策略优化")
print("=" * 60)

print("\n加载数据...")
customers = pd.read_csv(os.path.join(DATA_DIR, 'customers_info.csv'))
products = pd.read_csv(os.path.join(DATA_DIR, 'products.csv'))
behavior = pd.read_csv(os.path.join(DATA_DIR, 'user_behavior.csv'))
promotions = pd.read_csv(os.path.join(DATA_DIR, 'promotions.csv'))
locations = pd.read_csv(os.path.join(DATA_DIR, 'locations.csv'))

behavior['行为时间'] = pd.to_datetime(behavior['时间'])
behavior['日期'] = behavior['行为时间'].dt.date
behavior['月份'] = behavior['行为时间'].dt.month
behavior['年份'] = behavior['行为时间'].dt.year

df = behavior.merge(customers, on='用户ID', how='left')
df = df.merge(products, on='商品ID', how='left')

print(f"  用户信息: {customers.shape}")
print(f"  商品信息: {products.shape}")
print(f"  行为数据: {behavior.shape}")
print(f"  合并数据: {df.shape}")

# 1. 构建时间序列数据
print("\n构建时间序列数据...")

purchase_df = df[df['行为'] == '购买'].copy()
daily_sales = purchase_df.groupby(['日期', '商品类型']).size().unstack(fill_value=0)
daily_sales.index = pd.to_datetime(daily_sales.index)
daily_sales = daily_sales.resample('D').sum().fillna(0)

monthly_sales = purchase_df.groupby(['年份', '月份', '商品类型']).size().unstack(fill_value=0)
monthly_sales.index = pd.to_datetime(monthly_sales.index.map(lambda x: f'{int(x[0])}-{int(x[1]):02d}'))

print(f"  日销量数据: {daily_sales.shape}")
print(f"  月销量数据: {monthly_sales.shape}")

# 2. 时间序列预测模型
print("\n" + "=" * 60)
print("构建时间序列预测模型")
print("=" * 60)

top_categories = daily_sales.sum().nlargest(5).index.tolist()
print(f"  Top5商品类别: {top_categories}")

results = []
predictions_dict = {}

for category in top_categories:
    print(f"\n预测类别: {category}")
    
    series = daily_sales[category].values
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]
    
    # ARIMA模型
    print("  训练ARIMA模型...")
    try:
        arima_model = ARIMA(train, order=(2, 1, 2))
        arima_fitted = arima_model.fit()
        arima_pred = arima_fitted.forecast(steps=len(test))
        arima_pred = np.maximum(arima_pred, 0)
        arima_metrics = calc_metrics(test, arima_pred)
        print(f"    ARIMA - MAE: {arima_metrics['MAE']:.2f}, RMSE: {arima_metrics['RMSE']:.2f}, MAPE: {arima_metrics['MAPE']:.4f}")
    except:
        arima_pred = np.mean(train) * np.ones(len(test))
        arima_metrics = calc_metrics(test, arima_pred)
    
    # 指数平滑模型
    print("  训练指数平滑模型...")
    try:
        hw_model = ExponentialSmoothing(
            train, 
            trend='add', 
            seasonal='add', 
            seasonal_periods=7
        )
        hw_fitted = hw_model.fit()
        hw_pred = hw_fitted.forecast(steps=len(test))
        hw_pred = np.maximum(hw_pred, 0)
        hw_metrics = calc_metrics(test, hw_pred)
        print(f"    HW - MAE: {hw_metrics['MAE']:.2f}, RMSE: {hw_metrics['RMSE']:.2f}, MAPE: {hw_metrics['MAPE']:.4f}")
    except:
        hw_pred = np.mean(train) * np.ones(len(test))
        hw_metrics = calc_metrics(test, hw_pred)
    
    # 移动平均模型
    print("  训练移动平均模型...")
    window = 30
    ma_pred = []
    for i in range(len(test)):
        if train_size + i >= window:
            ma_pred.append(np.mean(series[train_size + i - window:train_size + i]))
        else:
            ma_pred.append(np.mean(train))
    ma_pred = np.array(ma_pred)
    ma_metrics = calc_metrics(test, ma_pred)
    print(f"    MA - MAE: {ma_metrics['MAE']:.2f}, RMSE: {ma_metrics['RMSE']:.2f}, MAPE: {ma_metrics['MAPE']:.4f}")
    
    # 选择最优模型
    models_metrics = {
        'ARIMA': arima_metrics['RMSE'],
        '指数平滑': hw_metrics['RMSE'],
        '移动平均': ma_metrics['RMSE']
    }
    best_model = min(models_metrics, key=models_metrics.get)
    
    results.append({
        '商品类别': category,
        'ARIMA_RMSE': arima_metrics['RMSE'],
        'ARIMA_MAPE': arima_metrics['MAPE'],
        '指数平滑_RMSE': hw_metrics['RMSE'],
        '指数平滑_MAPE': hw_metrics['MAPE'],
        '移动平均_RMSE': ma_metrics['RMSE'],
        '移动平均_MAPE': ma_metrics['MAPE'],
        '最优模型': best_model,
        '最优RMSE': models_metrics[best_model]
    })
    
    predictions_dict[category] = {
        'actual': test,
        'arima': arima_pred,
        'hw': hw_pred,
        'ma': ma_pred,
        'best': arima_pred if best_model == 'ARIMA' else (hw_pred if best_model == '指数平滑' else ma_pred)
    }

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, 'prediction_results.csv'), index=False, encoding='utf-8-sig')
print("\n预测结果已保存: prediction_results.csv")
print(results_df.to_string(index=False))

# 3. 未来销量预测
print("\n" + "=" * 60)
print("未来30天销量预测")
print("=" * 60)

future_predictions = {}
for category in top_categories:
    series = daily_sales[category].values
    try:
        model = ARIMA(series, order=(2, 1, 2))
        fitted = model.fit()
        future_pred = fitted.forecast(steps=30)
        future_pred = np.maximum(future_pred, 0)
        future_predictions[category] = future_pred
        print(f"  {category}: 未来30天总预测销量 = {future_pred.sum():.0f}")
    except:
        future_predictions[category] = np.mean(series) * np.ones(30)

# 4. 库存调配策略
print("\n" + "=" * 60)
print("制定库存调配策略")
print("=" * 60)

inventory_strategy = []
for category in top_categories:
    current_stock = products[products['商品类型'] == category]['库存'].sum()
    future_30d = future_predictions[category].sum()
    safety_stock = future_30d * 0.2
    reorder_point = future_30d / 30 * 7 + safety_stock
    
    if current_stock < reorder_point:
        action = '需要补货'
        reorder_qty = future_30d - current_stock + safety_stock
    else:
        action = '库存充足'
        reorder_qty = 0
    
    inventory_strategy.append({
        '商品类别': category,
        '当前库存': current_stock,
        '未来30天预测销量': future_30d,
        '安全库存': safety_stock,
        ' reorder点': reorder_point,
        '策略': action,
        '建议补货量': reorder_qty
    })

inventory_df = pd.DataFrame(inventory_strategy)
inventory_df.to_csv(os.path.join(OUTPUT_DIR, 'inventory_strategy.csv'), index=False, encoding='utf-8-sig')
print("\n库存策略已保存: inventory_strategy.csv")
print(inventory_df.to_string(index=False))

# 5. 差异化营销策略
print("\n" + "=" * 60)
print("制定差异化营销策略")
print("=" * 60)

user_profile = pd.read_csv(os.path.join(r'g:\A.lsit\08.建模\校赛-B 题  销售商品行为预测\问题3', 'user_profile_q3.csv'))

marketing_strategy = []
for segment in user_profile['用户分层'].unique():
    segment_users = user_profile[user_profile['用户分层'] == segment]
    
    if segment == '核心用户':
        strategy = 'VIP专属优惠 + 优先购买权 + 定制化推荐'
        discount = 0.15
        target_categories = segment_users['偏好商品类型'].mode().values[0] if len(segment_users['偏好商品类型'].mode()) > 0 else '全品类'
    elif segment == '活跃用户':
        strategy = '满减活动 + 积分翻倍 + 新品推荐'
        discount = 0.10
        target_categories = segment_users['偏好商品类型'].mode().values[0] if len(segment_users['偏好商品类型'].mode()) > 0 else '全品类'
    elif segment == '普通用户':
        strategy = '限时折扣 + 优惠券推送 + 热门商品推荐'
        discount = 0.08
        target_categories = '热销品类'
    elif segment == '低频用户':
        strategy = '唤醒优惠 + 首单立减 + 个性化推送'
        discount = 0.12
        target_categories = '历史浏览品类'
    else:
        strategy = '大额优惠券 + 短信唤醒 + 爆款推荐'
        discount = 0.20
        target_categories = '全品类'
    
    marketing_strategy.append({
        '用户分层': segment,
        '用户数量': len(segment_users),
        '平均消费金额': segment_users['总消费金额'].mean(),
        '策略类型': strategy,
        '优惠力度': discount,
        '目标品类': target_categories,
        '预期转化率提升': f'{discount * 100 + 5:.1f}%'
    })

marketing_df = pd.DataFrame(marketing_strategy)
marketing_df.to_csv(os.path.join(OUTPUT_DIR, 'marketing_strategy.csv'), index=False, encoding='utf-8-sig')
print("\n营销策略已保存: marketing_strategy.csv")
print(marketing_df.to_string(index=False))

print("\n" + "=" * 60)
print("问题4数据分析和模型构建完成！")
print("=" * 60)
