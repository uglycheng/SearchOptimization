class MDP:
    def __init__(self,s_a_dic,sa_r_dic,sa_t_dic,cal_r):
        self.action = s_a_dic # state_action_dictionary
        self.reward = sa_r_dic #state-action_reward_dictionary
        self.transition = sa_t_dic #state-action_transition-probility-dictionary
        if cal_r:
            self.calculate_reward = cal_r # self-defined function cal_r(s,a,target) to calculate reward
        else:
            self.calculate_reward = lambda s,a,t:self.reward[s]
        
    def value_iteration(self,threshold,max_iter,gamma,target=None):
        pre_s_v = {i:0 for i in self.action.keys()}
        i = 0
        diff = -float('inf')
        while (diff>threshold or i==0) and i<=max_iter:
            s_v = {i:0 for i in self.action.keys()}
            diff = -float('inf')
            for s in pre_s_v.keys():
                temp = [self.calculate_reward(s,a,target)+gamma*sum([self.transition[(s,a)][s_next]*pre_s_v[s_next] for s_next in self.transition[(s,a)]]) for a in self.action[s]]
                if temp:
                    s_v[s] = max(temp)
                else:
                    s_v[s] = self.calculate_reward(s,None,target)
                diff = max(abs(s_v[s]-pre_s_v[s]),diff)
            pre_s_v = dict(s_v)
            i+=1
        if diff<=threshold:
            print('Converged')
        else:
            print(diff)
            print('Reach Max Iteration Number')
        return s_v
    
    def policy_iteration(self,threshold,max_iter_improve,max_iter_evaluation,gamma,target=None):
        pre_policy = {s:set(self.action[s]).pop() for s in self.action if self.action[s]}
        i = 0
        plc_stable = False
        while not plc_stable and i<=max_iter_improve:
            plc_stable = True
            pre_s_v, converged = self.policy_evaluation(pre_policy,threshold,max_iter_evaluation,gamma,target)
            policy = {s:set(self.action[s]).pop() for s in self.action if self.action[s]}
            for s in pre_s_v.keys():
                if s not in policy:
                    continue
                temp = [(self.calculate_reward(s,a,target)+gamma*sum([self.transition[(s,a)][s_next]*pre_s_v[s_next] for s_next in self.transition[(s,a)]]),a) for a in self.action[s]]
                policy[s] = max(temp)[1]
                if policy[s]!= pre_policy[s]:
                    plc_stable = False
            pre_policy = dict(policy)
            i+=1
        
        if plc_stable:
            print('Policy Iteration Converged')
        else:
            print('Iteration Reach Max Iteration Number')
            
        return policy
    
    def policy_evaluation(self,pre_policy,threshold,max_iter_evaluation,gamma,target=None):
        pre_s_v = {i:0 for i in self.action.keys()}
        i = 0
        diff = -float('inf')
        while (diff>threshold or i==0) and i<=max_iter_evaluation:
            s_v = {i:0 for i in self.action.keys()}
            diff = -float('inf')
            for s in pre_s_v.keys():
                if s in pre_policy:
                    a = pre_policy[s]
                    s_v[s] = self.calculate_reward(s,a,target)+gamma*sum([self.transition[(s,a)][s_next]*pre_s_v[s_next] for s_next in self.transition[(s,a)]])
                else:
                    s_v[s] = self.calculate_reward(s,None,target)
                diff = max(abs(s_v[s]-pre_s_v[s]),diff)
            pre_s_v = dict(s_v)
            i+=1
        if diff<=threshold:
            print('Policy Evaluation Converged'+str(pre_policy))
            return s_v,True
        else:
            print(diff)
            print('Evaluation Reach Max Iteration Number'+str(pre_policy))
            return s_v,False
        
        
if __name__ == '__main__':
    action = {'A':set(['a1','a2']),'B':set(['b1','b2']),'C':set([])}
    reward = {'A':10,'B':10,'C':100}
    transtion = {('A','a1'):{'B':0.7,'C':0.3},
                ('A','a2'):{'B':0.3,'C':0.7},
                ('B','b1'):{'A':0.9,'C':0.1},
                ('B','b2'):{'A':0.1,'C':0.9},
                }

    mdp = MDP(action,reward,transtion,None)

    print('Value Iteration, gamma=0.1')
    s_v = mdp.value_iteration(0.00001,100,0.1)
    print(s_v)
    print('\n')

    print('Value Iteration, gamma=0.99')
    s_v = mdp.value_iteration(0.00001,100,0.99)
    print(s_v)
    print('\n')

    print('Policy Iteration, gamma=0.1')
    p = mdp.policy_iteration(0.00001,100,100,0.1)
    print(p)
    print('\n')

    print('Policy Iteration, gamma=0.99')
    p = mdp.policy_iteration(0.00001,100,100,0.99)
    print(p)
    print('\n')

