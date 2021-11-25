from numpy import polyfit, polyval, array, arange
from matplotlib.pyplot import plot,show,rc
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib
matplotlib.use('TkAgg')    # 解决版本兼容性问题

t=np.array([7,67,87,190,191])
C=np.array([0.045,0.087,0.087,0.095,0.092])
p=polyfit(t,C,2)
print(p)
time= np.linspace(7,191,num=20)
yhat = p[0]*time*time + p[1]*time + p[2]
plot(t, C, 'o', time,yhat ,'-')
show()
def f(x):
    return p[0]*x*x+p[1]*x+p[2]

h = 0
for i in range(0,5):
    R = pow(C[i]-f(t[i]),2)
    h = h + R
print("用二次多项式模型拟合的残差平方和为：",h)


