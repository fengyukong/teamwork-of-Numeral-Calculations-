import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

x=[0.5,0.6,0.7]
y=[-0.693147, -0.510826, -0.356675]
48
f = interpolate.interp1d(x, y ,kind='quadratic')
xnew = 0.65
ynew = f(xnew) # use interpolation function returned by `interp1d`
print(ynew)

print("=======================================分割线====================================")

x=[0.4,0.6,0.8]
y=[-0.916291,-0.510826,-0.223144]
f1 = interpolate.interp1d(x, y,kind = 'quadratic')
xnew = np.linspace(0.4,0.8,3)
plt.plot(x, y, 'o', xnew, f1(xnew), '--')
plt.legend(['data', 'quadratic', 'cubic','nearest'], loc = 'best')
plt.show()