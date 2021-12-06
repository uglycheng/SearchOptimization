from torch import nn
import torch
from collections import deque
import random
import copy

class Net(nn.Module):
    def __init__(self,shape,action_size,action_dim):
        super(Net, self).__init__()
        layers = []
        for i in range(1,len(shape)):
            layers.append(nn.Linear(shape[i-1],shape[i]))
            layers.append(nn.ReLU())
        layers.pop()
        self.net = nn.Sequential(*layers)
        self.embedding = nn.Embedding(action_size,action_dim)
    def forward(self,x):
        state = x[0]
        action = x[1]
        action_embedding = self.embedding(torch.as_tensor(action,dtype=torch.int))
        x = torch.cat((torch.as_tensor(state,dtype=torch.float),action_embedding),axis=-1)
        return self.net(x)
    
    def encode(self,state,action):
        action_embedding = self.embedding(torch.as_tensor(action,dtype=torch.int))
        return torch.cat((torch.as_tensor(state,dtype=torch.float),action_embedding),axis=-1)
    
class DQN:
    def __init__(self,net,memory_size,gamma,epsilon,actions):
        self.net = net
        self.net2 = copy.deepcopy(net)
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_space = list(actions)
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.mse = nn.MSELoss()
    
    def store(self,state,action,reward,nstate,terminal):
        self.memory.append((state,action,reward,nstate,terminal))
    
    def act(self,state):
        if torch.rand(1)>self.epsilon:
            acts = list(self.action_space)
            states = [state for i in range(len(acts))]
            optimal_act = self.action_space[torch.argmax(self.net([states,acts]))]
            return optimal_act
        else:
            return random.choice(self.action_space)
            
    def replay(self,batch_size):
        self.net2.load_state_dict(self.net.state_dict())
        batch = random.sample(self.memory,batch_size)
        states = [i[0] for i in batch]
        actions_ = [i[1] for i in batch]
        rewards = torch.as_tensor([i[2] for i in batch],dtype=torch.float)
        terminal = 1.0-torch.as_tensor([i[4] for i in batch],dtype=torch.float)
        nstates = []
        nacts = []
        acts = list(self.action_space)
        for i in range(len(batch)):
            nacts.extend(acts)
            nstates.extend([batch[i][3] for j in range(len(acts))])
        nqvalue = torch.amax(torch.reshape(self.net2([nstates,nacts]),(-1,len(acts))),dim=-1)
        labels = rewards+self.gamma*terminal*nqvalue
        predictions = self.net([states,actions_])
        loss = self.mse(torch.squeeze(predictions),torch.squeeze(labels))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



if __name__ =='__main__':
    from MechanicalArm import MechanicalArm
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import numpy as np

    target_point = [1.2,1.3]
    net = Net([4,16,1],9,2)
    dqn = DQN(net,2000,0.95,0.1,list(range(9)))
    seqs = []
    for e in range(10):
        env = MechanicalArm(1,random.randint(0,180),random.randint(0,180))
        env.set_target(target_point[0],target_point[1])
        seqs.append([])
        for time in range(5000):
            state = [env.theta1,env.theta2]
            action = dqn.act(state)
            env.act(action)
            seqs[-1].append([env.p1,env.p2])
            new_state = [env.theta1,env.theta2]
            reward = env.reward()
            
            if env.dis() < 0.1:
                terminal = True
            else:
                terminal = False
            dqn.store(state,action,reward,new_state,terminal)
            
            if terminal:
                print(time,'Finish')
                break
            if len(dqn.memory)>=500:
                dqn.replay(500)

    seq_gif = list(seqs[-1])
    for i in range(20):
        seq_gif.append(seq_gif[-1])
    
    x_data = [0,0,0]
    y_data = [0,0,0]
    L = 1
    fig, ax = plt.subplots()
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5,2)
    line, = ax.plot(0, 0)

    plt.scatter([target_point[0]],[target_point[1]],color='r')
    plt.scatter([0],[0],color='black')
    def animation_frame(i):

        x_data[1] = seq_gif[i][0][0]
        y_data[1] = seq_gif[i][0][1]
        x_data[2] = seq_gif[i][1][0]
        y_data[2] = seq_gif[i][1][1]

        line.set_xdata(x_data)
        line.set_ydata(y_data)
        return line, 

    animation = FuncAnimation(fig, func=animation_frame, frames=range(len(seq_gif)),interval=50)
    plt.show()