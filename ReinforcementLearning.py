import random
import copy
class Environment:
    def __init__(self,s_a_dic,sa_t_dic,cur_state,target):
        self.action = s_a_dic # state_action_dictionary
        self.transition = sa_t_dic #state-action_transition-probility-dictionary
        self.cur_state = cur_state
        self.target = target
        self.transition_presum = {}
        for key,v in self.transition.items():
            s = 0
            self.transition_presum[key] = {}
            for s_next,t in v.items():
                self.transition_presum[key][s_next] = [s,s+t]
                s = s+t
            
    def if_terminate(self):
        if self.cur_state in self.target:
            return True
        else:
            return False
    
    
    def step(self,a):
        if self.if_terminate() or a not in self.action[self.cur_state]:
            return False
        r = random.uniform(0,1)
        for s,interval in self.transition_presum[(self.cur_state,a)].items():
            if interval[0]<=r<interval[1] or (r==1 and interval[1]==1):
                self.cur_state = s
                return True
                
class DiscreteRL:
    def __init__(self,reward,cal_r,env):
        self.reward = reward
        if cal_r:
            self.calculate_reward = cal_r # a function cal_r(s,a,target) to calculate reward
        else:
            self.calculate_reward = lambda s,a,t:self.reward[s]
        self.env = copy.deepcopy(env)
            
    def mc_policy(self,policy,gamma,max_iter=100):
        s_G = {i:[] for i in self.env.action}
        s_U = {i:0 for i in self.env.action}
        i = 0
        while i<=max_iter:
            s = random.choice(list(s_G.keys()))
            seq = [s]
            self.env.cur_state = s
            while not self.env.if_terminate():
                success = self.env.step(policy[seq[-1]])
                if not success:
                    raise ValueError('Invalid Policy')
                seq.append(self.env.cur_state)
            G = self.calculate_reward(seq[-1],None,self.env.target)
            s_G[seq[-1]].append(G)
            for j in range(2,len(seq)+1):
                G = gamma*G + self.calculate_reward(seq[-j],policy[seq[-j]],self.env.target)
                s_G[seq[-j]].append(G)
            for s in s_G:
                if len(s_G[s]) > 0:
                    s_U[s] = sum(s_G[s])/len(s_G[s])
            i += 1
        return s_U
        
        
    def TD_policy(self,policy,gamma,max_iter=100):
        s_U = {i:0 for i in self.env.action}
        for tars in self.env.target:
            s_U[tars] = self.reward[tars]
        s_N = {i:0 for i in self.env.action}
        i = 0
        while i<=max_iter:
            s = random.choice(list(s_U.keys()))
            s_N[s] += 1
            self.env.cur_state = s
            while not self.env.if_terminate():
                self.env.step(policy[s])
                s_next = self.env.cur_state
                s_N[s_next] += 1
                # alpha = 10/(9+s_N[s])
                s_U[s] = s_U[s] + (10/(9+s_N[s])) * (self.calculate_reward(s,policy[s],self.env.target)+gamma*s_U[s_next]-s_U[s])
                s = s_next
            i+=1
        return s_U
    
    def Q_learning(self,gamma,eps,max_iter=100):
        s_Q = {}
        for s,a_set in self.env.action.items():
            for a in a_set:
                s_Q[(s,a)] = 0
        for tars in self.env.target:
            s_Q[tars] = self.reward[tars]
        s_N = {i:0 for i in self.env.action}
        i = 0
        while i<=max_iter:
            s = random.choice(list(self.env.action.keys()))
            s_N[s] += 1
            self.env.cur_state = s
            while not self.env.if_terminate():
#                 print(s)
                a = self.Q_pick_action(eps,s,s_Q)
                self.env.step(policy[s])
                s_next = self.env.cur_state
                s_N[s_next] += 1
                if s_next in self.env.target:
                    max_Q_next = s_Q[s_next]
                else:
                    max_Q_next = max([s_Q[(s_next,a_next)] for a_next in self.env.action[s_next]])
#                 alpha = 0.1
                alpha = (10/(9+s_N[s]))
                s_Q[(s,a)] = s_Q[(s,a)] + alpha * (self.calculate_reward(s,a,self.env.target)+gamma*max_Q_next-s_Q[(s,a)])
                s = s_next
            i+=1
#             print(s_Q)
        return s_Q
    
    def Q_pick_action(self,eps,s,s_Q):
        r = random.uniform(0,1)
        if r<eps:
            return random.choice(list(self.env.action[s]))
        else:
            max_Q = -float('inf')
            max_a = ''
            for a in self.env.action[s]:
                Q = s_Q[(s,a)]
                if Q>max_Q:
                    max_Q = Q
                    max_a = a
                elif Q==max_Q:
                    max_Q,max_a = random.choice([(max_Q,max_a),(Q,a)])
            return max_a


# class ContinuousRL

#     def DQN


if __name__ == "__main__":
    action = {'A':set(['a1','a2']),'B':set(['b2','b1']),'C':set([])}
    reward = {'A':10,'B':10,'C':100}
    transtion = {('A','a1'):{'B':0.7,'C':0.3},
                ('A','a2'):{'B':0.3,'C':0.7},
                ('B','b1'):{'A':0.9,'C':0.1},
                ('B','b2'):{'A':0.1,'C':0.9},
                }

    env = Environment(action,transtion,'A',['C'])
    rl = DiscreteRL(reward,None,env)

    policy = {'A': 'a1', 'B': 'b1'}
    print('Monte Carlo Policy Evaluation, gamma=0.99, Policy: '+str(policy))
    print(rl.mc_policy(policy,0.99,10000))
    print('\n')

    policy = {'A': 'a1', 'B': 'b1'}
    print('Temporal Difference Policy Evaluation, gamma=0.99, Policy: '+str(policy))
    print(rl.TD_policy(policy,0.99,100000))
    print('\n')


    print('Q Learning, gamma=0.99, greedy=0.15')
    print(rl.Q_learning(0.99,0.15,200000))