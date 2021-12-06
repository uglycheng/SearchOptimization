import numpy as np

class MechanicalArm:
    def __init__(self,L,theta1,theta2):
        self.L = L
        self.theta1 = theta1%360
        self.theta2 = theta2%360
        self.p1,self.p2 = self.cal_pos(theta1,theta2)
        element = [-10,0,10]
        self.action = {i*3+j:[element[i],element[j]] for i in range(3) for j in range(3)}
    
    def set_target(self,x,y):
        self.tar = [x,y]
        
    def cal_pos(self,theta1,theta2):
        p1 = [self.L*np.cos(2*np.pi*theta1/360),self.L*np.sin(2*np.pi*theta1/360)]
        p2 = [p1[0]+self.L*np.cos(2*np.pi*(theta1+theta2)/360),p1[1]+self.L*np.sin(2*np.pi*(theta1+theta2)/360)]
        return p1,p2
    
    def act(self,a):
        theta_change = self.action[a]
        self.theta1 += theta_change[0]
        self.theta2 += theta_change[1]
        self.theta1 = self.theta1%360
        self.theta2 = self.theta2%360
        self.p1,self.p2 = self.cal_pos(self.theta1,self.theta2)
        return 
    
    def reward(self):
        return 1/(((self.p2[0]-self.tar[0])**2+(self.p2[1]-self.tar[1])**2)+0.001)
    
    def dis(self):
        return ((self.p2[0]-self.tar[0])**2+(self.p2[1]-self.tar[1])**2)**0.5
    