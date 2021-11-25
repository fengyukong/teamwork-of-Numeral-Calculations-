from numpy import polyfit, polyval, array, arange
from matplotlib.pyplot import plot,show,rc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
matplotlib.use('TkAgg')    # 解决版本兼容性问题

year = np.array([35,45,55,65,75])
blood_pressure = np.array([114,124,148,158,166])
X=1./year
Y=np.log(blood_pressure)
p=polyfit(X,Y,1)
# P返回的是一个列表,利用的是polyfit自带的多项式拟合功能，最高项次数为“1”，
# print(p)
print("b=",p[0])
print("k",p[1])
print("a=",math.exp(p[1]))

time= np.linspace(35,75,num=20)
print("time:",time)
yhat = np.exp(p[1])*np.exp(p[0]/time)  # 即函数表达式
print("yhat:",yhat)
plot(year, blood_pressure, 'o', time,yhat ,'-')
# 前三个参数可以画出“o”标记的散点图  后面的time 是自变量 yhat 是因变量
plt.title("blood pressure relationship towards age")
show()
