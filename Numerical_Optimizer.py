import torch
import numpy as np

class GradientOptimizer:
    def __init__(self,func):
        self.func = func

    def gradient_descent(self,x0,lr,max_iter=100):
        y_history = []
        x_history = []
        x = torch.tensor([float(i) for i in x0],requires_grad=True)
#         lr = torch.tensor(lr)
        for i in range(max_iter+1):
            x_history.append([float(i) for i in x])
            y = self.func(x)
            y_history.append(float(y))
            g = torch.autograd.grad(y,x)[0]
            x = (x-lr*g).clone().detach().requires_grad_(True)
        return x_history ,y_history

    def newton_descent(self,x0,lr,max_iter=100):
        y_history = []
        x_history = []
        x = torch.tensor([float(i) for i in x0],requires_grad=True)
        for i in range(max_iter+1):
            x_history.append([float(i) for i in x])
            y = self.func(x)
            y_history.append(float(y))
            h_inv = torch.autograd.functional.hessian(self.func,x).inverse()
            g = torch.autograd.grad(y,x)[0]
            x = (x-lr*torch.matmul(h_inv,g)).clone().detach().requires_grad_(True)
        return x_history ,y_history

    def conjugate_descent(self,x0,lr=None,max_iter = 100):
        epsilon = 1e-10 # prevent divide 0
        y_history = []
        x_history = []
        x = torch.tensor([float(i) for i in x0],requires_grad=True)
        
        x_history.append([float(i) for i in x])
        y = self.func(x)
        y_history.append(float(y))
        Q = torch.autograd.functional.hessian(self.func,x)
        p = torch.autograd.grad(y,x)[0]
        alpha = -torch.matmul(p,p)/(torch.matmul(p,torch.matmul(Q,p))+epsilon)
        x = (x+alpha*p).requires_grad_(True)
        p_pre = p
        
        
        for i in range(1,max_iter+1):
            x_history.append([float(i) for i in x])
            y = self.func(x)
            y_history.append(float(y))
            Q = torch.autograd.functional.hessian(self.func,x)
            r = torch.autograd.grad(y,x)[0]
            beta = torch.matmul(p_pre, torch.matmul(Q,r))/(torch.matmul(p_pre, torch.matmul(Q,p_pre))+epsilon)
            p = -r+beta*p_pre
            alpha = -torch.matmul(r,p)/(torch.matmul(p,torch.matmul(Q,p))+epsilon)
            x = (x+alpha*p).requires_grad_(True)
            p_pre = p
        return x_history ,y_history 


class StochasticOptimizer:
    def __init__(self,func):
        self.func = func

    def simulated_annealing(self,x0,T0,T_decay_method='fast',gamma=0.8,seed=29,max_iter=100):
        if T_decay_method not in set(['fast','exponential','log']):
            print('T_decay_method must be one of fast, exponential,log')
            return [],[]
        if T_decay_method=='exponential':
            if not (0<gamma<1):
                print('gamma must be bigger than 0 and smaller than 1')
                return [],[]
            
            Tk = T0/gamma
        np.random.seed(seed)
        cov = np.identity(len(x0))
        x = np.array(x0)
        y_history = []
        x_history = []
        for k in range(1,max_iter+2):
            x_history.append([float(i) for i in x])
            y = self.func(x)
            y_history.append(float(y))
            newx = np.random.multivariate_normal(x,cov)
            E = self.func(newx)-y_history[-1]
            if T_decay_method=='fast':
                Tk = T0/k
            elif T_decay_method=='exponential':
                Tk = gamma*Tk
            elif T_decay_method=='log':
                Tk = T0*np.log(2)/np.log(k+1)
                
            if E<0 or self.if_sa_accecpt(Tk,E):
                x_history.append(newx)
                x = newx
            else:
                x_history.append(x)
        return x_history,y_history
    
    def if_sa_accecpt(self,T,E):
        p_accept = np.exp(-E/T)
        sample = np.random.uniform(0,1)
        if sample <= p_accept:
            return True
        else:
            return False

    
    def cross_entropy(self,x0,num_sample,ratio,threshold=0,seed=29,max_iter=100):
        if not (0<ratio<=1):
            print('ratio should be bigger than 0 and less or equal to 1')
        np.random.seed(seed)
        u = np.array(x0)
        y_history = [self.func(u)]
        x_history = [u]
        cov = np.identity(len(x0))
        for i in range(max_iter):
            samples = []
            for j in range(num_sample):
                samples.append(np.random.multivariate_normal(u,cov))
            y_history.append(sum([self.func(s) for s in samples])/num_sample)
            samples.sort(key=lambda x:self.func(x))
            u,cov = self.update_gauss(samples[:int(num_sample*ratio)],threshold)
            x_history.append(u)
        return x_history,y_history
     
    def update_gauss(self,E,threshold):
        s = len(E)
        u = np.sum(np.array(E),axis=0)/s
        temp = np.array(E) - u
        cov = np.sum(np.matmul(temp[:,:,None],temp[:,None,:]),axis=0)/s
        if np.sum(np.abs(cov))<=threshold:
            cov = np.identity(len(E[0]))
        return u,cov 


    def search_gradient(self,x0,num_sample,lr,seed=29,max_iter=100):
        np.random.seed(seed)
        u = np.array(x0)
        y_history = [self.func(u)]
        x_history = [u]
        cov = np.identity(len(x0))
        last_cov = np.identity(len(x0))
        fix_cov = False
        for i in range(max_iter):
            gs_u = []
            gs_cov = []
            vs = []
            try:
                inv_cov = np.inalg.inv(cov)
            except:
                break
            for s in range(num_sample):
                x = np.random.multivariate_normal(u,cov)
                v = self.func(x)
                vs.append(v)
                g_u,g_cov = self.g_log_guass(x,u,inv_cov)
                gs_u.append(g_u*v)
                gs_cov.append(g_cov*v)
            gs_u = np.mean(gs_u,axis=0)
            gs_cov = np.mean(gs_cov,axis=0)
            gs_u_norm = (np.sum(gs_u**2))**0.5
            gs_cov_norm = (np.sum(gs_u_norm**2))**0.5
            gs_u /= gs_u_norm
            gs_cov /= gs_cov_norm


            x_history.append(np.array(u)-lr*gs_u)
            y_history.append(sum(vs)/num_sample)
            last_cov = np.array(cov)
            cov = last_cov-lr*gs_cov

        cov = np.array(last_cov)
        inv_cov = np.linalg.inv(cov)
        for i2 in range(i,max_iter):
            gs_u = []
            vs = []
            u = x_history[-1]
            for s in range(num_sample):
                x = np.random.multivariate_normal(u,cov)
                v = self.func(x)
                vs.append(v)
                g_u = self.g_log_guass_u(x,u,inv_cov)
                gs_u.append(g_u*v)
            gs_u = np.mean(gs_u,axis=0)
            gs_u_norm = (np.sum(gs_u**2))**0.5
            gs_u /= gs_u_norm
            x_history.append(np.array(u)-lr*gs_u)
            y_history.append(sum(vs)/num_sample)
        return x_history,y_history
    
    def g_log_guass(self,x,u,inv_cov):
        g_u = np.matmul(inv_cov,np.array(x)-np.array(u))
        g_cov = -0.5*inv_cov+0.5* np.matmul(inv_cov,np.matmul((x-u)[:,None]*(x-u)[None,:],inv_cov))
        return g_u,g_cov

    def g_log_guass_u(self,x,u,inv_cov):
        g_u = np.matmul(inv_cov,np.array(x)-np.array(u))
        return g_u