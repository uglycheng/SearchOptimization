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
Import **GradientOptimizer** defined in Numerical_Optimizer.py. To initialize a instance of this class, you only need to tell the target function to be optimized to the class. It works better if the operations in the target function are all Pytorch operations. This class has three methods:
- **gradient_descent**(x0,lr,max_iter=100)
- **newton_descent**(x0,lr,max_iter=100)
- **conjugate_descent**(x0,lr=None,max_iter=100)


All of them has three parameters:
- **x0**: initial point, it should have the same format of your target function's input
- **lr**: learning rate, float.
- **max_iter**: max number of iterations, int. The optimization process will stop after max number of iterations even if it doesn't converge.


All of them will return 2 lists:
- **x_history**: the target function's variable values after each descent step.
- **y_history**: the target function's values after each descent step.

Here is a plot of the three methods' optimization process for 
<img src="http://latex.codecogs.com/gif.latex? f(x_1,x_2,x_3,x_4,x_5) = 9x_1^{2} + 2x_2^{2} + 6x_3^{2} + 5x_4^{2} + 6x_5^{2}" style="border:none;">
![PNG](./figs/gradient_based.png)

## Stochastic Methods
Import **StochasticOptimizer** from Numerical_Optimizer.py. The initialization is the same as the GradientOptimizer except the target function doesn't have to be Pytorch-based functions. This class has three methods:
- **simulated_annealing**(x0,T0,T_decay_method='fast',gamma=0.8,seed=29,max_iter=100)
  - **x0**: initial point, it should have the same format of your target function's input 
  - **T0**: initial temperature, float.
  - **T_decay_method**: temperature decay method. It should be one of 'fast','exponential','log'
  - **gamma**: decay constant if your T_decay_method is 'exponential', float.
  - **seed**: sample random seed, int.
  - **max_iter**: max number of iterations, int. The optimization process will stop after max number of iterations even if it doesn't converge.
- **cross_entropy**(x0,num_sample,ratio,threshold=0,seed=29,max_iter=100)
  - **x0**: initial point, it should have the same format of your target function's input 
  - **num_sample**: number of samples in each iteration, int
  - **ratio**: ratio of samples used to update the distribution, float
  - **threshold**: float. During the search process, the covariance matrix may become a almost zero matrix. To solve this problem, when sum of the matrix elements' absolute values is no larger than threshold, we set the covariance matrix to be the identity matrix.
  - **seed**: sample random seed, int.
  - **max_iter**: max number of iterations, int. The optimization process will stop after max number of iterations even if it doesn't converge. 
- **search_gradient**(x0,num_sample,lr,seed=29,max_iter=100)
  - **x0**: initial point, it should have the same format of your target function's input 
  - **num_sample**: number of samples in each iteration, int
  - **lr**: learning rate, float.
  - **seed**: sample random seed, int.
  - **max_iter**: max number of iterations, int. The optimization process will stop after max number of iterations even if it doesn't converge.  

All of them will return 2 lists:
- **x_history**: the target function's variable values after each descent step.
- **y_history**: the target function's values after each descent step.

Here is a plot of the three methods' optimization process for 
<img src="http://latex.codecogs.com/gif.latex? f(x_1,x_2) = x_1^{2} + 3x_2^{2}" style="border:none;">
![PNG](./figs/stochastic.png)

## A-Star
Import **A_star** from ClassicalSearch.py. This class implements A-star search on a 2D map. To initialize a instance, you should input a matrix(2D list) as the maze containing 0 and 1 where 0 represents avaliable locations 1 represents unavailable locations. This class only support 4 actions which are moving left, right, up, down. Use the method **search** to find the best path.
- **search**(x0,target,h):
  - **x0**: a list with the length 2 representing the start point.
  - **target**: a list with the length 2 representing the target point.
  - **h**: your self defined heuristic function.

## Minimax
Import **Minimax** from GameSearch.py. This class implements Minimax search on the [Stone Game](https://leetcode-cn.com/problems/stone-game/). Import Game from GameSearch.py as the game simulator. Note that using dynamic programming, this game's winner can be predicted easily. Using this Minimax class is not the best way to solve the problem, this class aims to show how the algorithm works.

- **Game**: Initialize the class with the list of the stones. It has two methods and 4 useful attributes:
  -**step**(player,head_tail): player should be either 'A' or 'B' indicating the player to take a stone. head_tail should be either 'h' or 't' indicating which stone to take
  - **undo**() undo the last step.
  - **A** player A's score.
  - **B** player B's score.
  - **head** current index of the head of the stone list.
  - **tail** current index of the tail of the stone list.
- **Node**: Tree node used for minimax search. It has three attributes:
  - **children**: a list of instances of Node class representing its children nodes.
  - **player**: current player to take the next stone
  - **score**: current player the score
- **Minimax**: Initialize the class with maximum tree layers **layer**(int), which **player**(either 'A' or 'B', 'A' takes the first turn) you are and the game simulator **game**. After initialization, the instance will have a attribute **node**(Instance of class Node) denoting your status at the beginning of the game. The method **minimax** take the starting node as the root and perform the minimax algorithm. To search from the beginning of the game, use the attribute node as the method minimax's input.

