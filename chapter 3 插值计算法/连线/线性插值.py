from scipy import interpolate
x=[0.6,0.7]
y=[-0.510826,-0.356675]
f = interpolate.interp1d(x, y ,kind='linear')
xnew = 0.65
ynew = f(xnew) # use interpolation function returned by `interp1d`
print(ynew)