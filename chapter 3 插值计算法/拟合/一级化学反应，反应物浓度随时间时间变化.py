from numpy import polyfit, polyval, array, arange
from matplotlib.pyplot import plot,show,rc
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib
matplotlib.use('TkAgg')    # 解决版本兼容性问题

t=np.array([0,2,4,6,8])

b=np.array([math.log(12),math.log(6.8),math.log(4.2),math.log(3.2),math.log(3.0)])
p=polyfit(t,b,1)
print(p)
time= np.linspace(0,8,num=20)
yhat = p[0]*time+p[1]
plot(t, b, 'o', time,yhat ,'-')

show()