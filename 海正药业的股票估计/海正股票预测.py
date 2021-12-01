import pandas as pd
import numpy as np
import statsmodels #时间序列
import seaborn as sns
import matplotlib.pylab as plt
from scipy import  stats
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.api import qqplot


matplotlib.use('TkAgg')    # 解决版本兼容性问题

# 1.数据预处理
Sentiment = pd.read_csv('600267四个月股票收盘价格3.csv', index_col='date', parse_dates=['date'])
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
ts_train = Sentiment.iloc[:n_train]['final_price']
ts_test = Sentiment.iloc[:n_forecast]['final_price']

sentiment_short = Sentiment.loc['2021-07-05':'2021-11-30']
 # print(sentiment_short
i = Sentiment.loc["2021-11-11":"2021-11-30"]
sentiment_short.plot(figsize=(12, 8))
plt.title("final_price")
plt.legend(bbox_to_anchor=(0.4, 0.8)) # legend: 图例 bbox_to_anchor 控制图摆放的位置
sns.despine()  # 隐藏图的右边和上边的边框线
plt.show()

# 2.时间序列的差分d——将序列平稳化
sentiment_short['diff_1'] = sentiment_short['final_price'].diff(1)
# 1个时间间隔，一阶差分，再一次是二阶差分
sentiment_short['diff_2'] = sentiment_short['diff_1'].diff(1)

sentiment_short.plot(subplots=True, figsize=(18, 12)) # 这里画了图


dta = sentiment_short.diff(1)

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111) # 将图画在第一行的第一列
diff1 = dta.diff(1)
diff1.plot(ax=ax1)  # 这里一张

fig = plt.figure(figsize=(12, 8))
ax2 = fig.add_subplot(111)
diff2 = sentiment_short.diff(2)
diff2.plot(ax=ax2)   # 这里又一张

plt.show()
# 上面代码基本没什么问题

del sentiment_short['diff_2']
del sentiment_short['diff_1']
print(sentiment_short.head())
print(type(sentiment_short))
# 3.1.分别画出ACF(自相关)和PACF（偏自相关）图像
fig = plt.figure(figsize=(12,8))
#画ACF
ax1 = fig.add_subplot(211)

sentiment_short = sentiment_short.dropna()

fig = sm.graphics.tsa.plot_acf(sentiment_short, lags=34,ax=ax1)#lags表示滞后的阶数
ax1.xaxis.set_ticks_position('bottom')
fig.tight_layout();
#画PACF
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(sentiment_short, lags=34, ax=ax2)
ax2.xaxis.set_ticks_position('bottom')
fig.tight_layout();


def tsplot(y, lags=None, title='', figsize=(14, 8)):
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0))    #折线图
    hist_ax = plt.subplot2grid(layout, (0, 1))  # 直方图
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    y.plot(ax=ts_ax)
    ts_ax.set_title(title)
    y.plot(ax=hist_ax, kind='hist', bins=25)
    hist_ax.set_title('Histogram')
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)   #acf 图
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)    # pacf 图
    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    plt.tight_layout()
    return ts_ax, acf_ax, pacf_ax


tsplot(sentiment_short, title='Final_price', lags=34)
plt.show()
# 上面那个直方图还可以再润色亿下

# 4.建立模型——参数选择
# 遍历，寻找适宜的参数
import itertools

p_min = 0
d_min = 0
q_min = 0
p_max = 4
d_max = 0
q_max = 4

# Initialize a DataFrame to store the results,，以BIC准则
results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min, p_max + 1)],
                           columns=['MA{}'.format(i) for i in range(q_min, q_max + 1)])

for p, d, q in itertools.product(range(p_min, p_max + 1),
                                 range(d_min, d_max + 1),
                                 range(q_min, q_max + 1)):
    if p == 0 and d == 0 and q == 0:
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
        continue

    try:
        model = sm.tsa.ARIMA(ts_train, order=(p, d, q),
                             # enforce_stationarity=False,
                             # enforce_invertibility=False,
                             )
        results = model.fit()
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
    except:
        continue
results_bic = results_bic[results_bic.columns].astype(float)

#画出热度图
fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(results_bic,
                 mask=results_bic.isnull(),
                 ax=ax,
                 annot=True,
                 fmt='.2f',
                 )
ax.set_title('BIC')
plt.show()


# 模型评价准则
train_results = sm.tsa.arma_order_select_ic(ts_train, ic=['aic', 'bic'], trend='n', max_ar=4, max_ma=4)

print('AIC', train_results.aic_min_order)
print('BIC', train_results.bic_min_order)

# 确定参数
arima112 = sm.tsa.ARIMA(ts_train, order=(1, 1, 2)).fit()  # (p,d,q)
model_results = sm.tsa.ARIMA(ts_train, order=(1, 2,0)).fit()

# 如果上面的结果不一致
# arma_mod20 = sm.tsa.ARMA(sentiment_short,(3,4)).fit()
# print(arma_mod20.aic,arma_mod20.bic,arma_mod20.hqic)
# arma_mod30 = sm.tsa.ARMA(sentiment_short,(1,2)).fit()
# print(arma_mod30.aic,arma_mod30.bic,arma_mod30.hqic)

'''
对ARMA(1,2)模型所产生的残差做自相关图
'''
resid =model_results.resid#残差
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=34, ax=ax2)
# D-W 检验
print(sm.stats.durbin_watson(model_results.resid.values))

# 观察是否符合正态分布，QQ图
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)

# Ljung-Box检验（白噪声检验）
r,q,p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
data = np.c_[range(1,20), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))

#6.模型预测 最后的最后！
predict_sunspots = model_results.predict("2021-11-10","2021-11-19", dynamic=True,type="levels")

print(predict_sunspots)
fig,ax = plt.subplots(figsize=(12, 8))
ax = Sentiment.loc['2021-7-02':].plot(ax=ax)
predict_sunspots.plot(ax=ax)
plt.show()