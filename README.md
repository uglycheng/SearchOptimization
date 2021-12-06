# SearchOptimization
This is a course project of UCSD CSE257 with Prof.Sicun Gao. It includes implementation of following methods:
- **Numerical Optimizers**
  - Gradient based methods: Gradient Descent, Newtown Descent, Conjugate Descent
  - Stochastic methods: Simulated Annealing, Cross Entropy, Search Gradient
- **A-Star Search**
- **Minimax Search**
- **Markov Decision Process**
  - Value Iteration
  - Policy Iteration
- **Reinforcement Learning**
  - Monte Carlo Policy Evaluation
  - Temporal Difference Policy Evaluation
  - Tabular Q learning
  - Deep Q Learning


## Gradient Based Methods:
Use GradientOptimizer defined in Numerical_Optimizer.py. To initialize a instance of this class, you only need to tell the target function to be optimized to the class. The operations in the target function must be Pytorch operations. This class has three methods:
- **gradient_descent**(x0,lr,max_iter=100)
- **newton_descent**(x0,lr,max_iter=100)
- **conjugate_descent**(x0,lr=None,max_iter=100)


All of them has three parameters:
- x0: initial point, it should have the same format of your target function's input
- lr: learning rate, float.
- max_iter: max number of iterations, int. The optimization process will stop after max number of iterations even if it doesn't converge.


All of them will return 2 lists:
- x_history: the target function's variable values after each descent step.
- y_history: the target function's values after each descent step.
