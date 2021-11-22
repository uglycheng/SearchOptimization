from Numerical_Optimizer import GradientOptimizer
import torch
import matplotlib.pyplot as plt

def gd_func(x):
    return sum((x**2)*torch.tensor([1,3],dtype=torch.float))

optimizer = GradientOptimizer(gd_func)
x,y = optimizer.gradient_descent([10.0,10.0],0.01)
plt.plot(list(range(len(y))),y,label='gradient')

x,y = optimizer.newton_descent([10.0,10.0],1)
plt.plot(list(range(len(y))),y,label='newton')

x,y = optimizer.conjugate_descent([10.0,10.0],max_iter=100)
plt.plot(list(range(len(y))),y,label='conjugate')
plt.legend()
plt.show()