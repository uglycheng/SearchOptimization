from Numerical_Optimizer import StochasticOptimizer
import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return sum((x**2)*np.array([1.0,3.0]))

optimizer = StochasticOptimizer(func)

x,y = optimizer.simulated_annealing([10.0,10.0],1000,max_iter=500,T_decay_method='log')
plt.plot(list(range(len(y))),y,label='SA-log')
x,y = optimizer.simulated_annealing([10.0,10.0],1000,max_iter=500,T_decay_method='fast')
plt.plot(list(range(len(y))),y,label='SA-fast')
x,y = optimizer.simulated_annealing([10.0,10.0],1000,max_iter=500,T_decay_method='exponential')
plt.plot(list(range(len(y))),y,label='SA-exponential')


x,y = optimizer.cross_entropy([10.0,10.0],100,0.2,threshold=0.1,max_iter=500)
plt.plot(list(range(len(y))),y,label='CE')

x,y = optimizer.search_gradient([10.0,10.0],100,0.1,max_iter=500)
plt.plot(list(range(len(y))),y,label='SG')

plt.legend()
plt.show()