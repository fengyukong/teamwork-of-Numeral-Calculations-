def myLagrange(x,y,a):
    ans=0.0
    for i in range(len(y)):
        t=y[i]
        for j in range(len(y)):
            if i !=j:
                t*=(a-x[j])/(x[i]-x[j])
        ans +=t
    return ans

x=[0.4,0.5, 0.6, 0.7 ,0.8]
y=[-0.916291 ,   -0.693147 ,  -0.510826  , -0.356675 ,  -0.223144]

print(myLagrange (x,y,0.65))