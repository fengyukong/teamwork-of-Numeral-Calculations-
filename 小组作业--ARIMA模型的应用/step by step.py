#美国消费者信心指数的预测
import pandas as pd
import numpy as np
import statsmodels #时间序列
import seaborn as sns
import matplotlib.pylab as plt
from scipy import  stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')    # 解决版本兼容性问题

# 1.数据预处理
Sentiment = pd.read_csv('data_test.csv', index_col='date', parse_dates=['date'])
# index_col=0
# parse_dates=[0]
#print(Sentiment)
print(Sentiment.head())  # 这个只返回头五个行数据
# 切分为测试数据和训练数据
print(Sentiment.shape) # 返回这个表的行列数（7，2）是一个元组
n_sample = Sentiment.shape[0]  # 获取表的行数

n_train = int(0.95 * n_sample) + 1  # int() 取整数位（不四舍五入）
# 相当于置信度是95%
n_forecast = n_sample - n_train
# 剩下的拿来做样本预测检验成果
ts_train = Sentiment.iloc[:n_train]['confidence']
ts_test = Sentiment.iloc[:n_forecast]['confidence']

sentiment_short = Sentiment.loc['2001-02-02':'2001-06-02']
 # print(sentiment_short)
sentiment_short.plot(figsize=(12, 8))
plt.title("Consumer Sentiment")
plt.legend(bbox_to_anchor=(0.4, 0.8)) # legend: 图例 bbox_to_anchor 控制图摆放的位置
sns.despine()  # 隐藏图的右边和上边的边框线
plt.show()

# 2.时间序列的差分d——将序列平稳化
sentiment_short['diff_1'] = sentiment_short['confidence'].diff(1)
# 1个时间间隔，一阶差分，再一次是二阶差分
sentiment_short['diff_2'] = sentiment_short['diff_1'].diff(1)

sentiment_short.plot(subplots=True, figsize=(18, 12))

sentiment_short = sentiment_short.diff(1)
dta = sentiment_short.diff(1)

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111)
diff1 = sentiment_short.diff(1)
diff1.plot(ax=ax1)

fig = plt.figure(figsize=(12, 8))
ax2 = fig.add_subplot(111)
diff2 = dta.diff(2)
diff2.plot(ax=ax2)

plt.show()
