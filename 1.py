import numpy as np
x=np.array([[1,1],[2,3],[3,3]])
y=np.array([3.0,7.0,8.8]).reshape(3,-1)
l=0.1
def cal(x,y,l):
    I=np.identity(2)
    Inverse=np.linalg.inv(np.matmul(x.T,x)+2*I*l)
    vec=np.matmul(np.matmul(Inverse,x.T),y)
    norm=np.linalg.norm(vec)
    print(f"{vec}, norm is {norm}")

cal(x,y,0.1)
cal(x,y,0.25)
cal(x,y,0.5)