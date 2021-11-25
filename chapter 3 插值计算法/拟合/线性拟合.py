from numpy import polyfit, polyval, array, arange
from matplotlib.pyplot import plot,show,rc
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')    # 解决版本兼容性问题

t=np.array([5,10,15,20,25,30,35,40,45,50,55])
y=np.array([1.2,2.16,2.86,3.44,3.87,4.15,4.37,4.51,4.58,4.62,4.64])
X=1./t
Y=np.log(y)
p=polyfit(X,Y,1)
print(p)
time= np.linspace(5,55,num=20)
yhat = np.exp(p[1])*np.exp(p[0]/time)  # 确定函数很重要1
plot(t, y, 'o', time,yhat ,'-')
plt.title('chemical products concentration changes with time ')  # 给图添加一个标题
show()