import numpy as np
from scipy import optimize

class NelsonSiegelModels(object):
    def __init__(self, size):
        self.r         = np.zeros(size)
        self.df        = np.zeros(size)
            
    def Load_CF_Prices(self,time_, Cashflows_, Prices_):
        self.time      = np.asarray(time_)
        self.Cashflows = np.asarray(Cashflows_)
        self.Prices    = np.asarray(Prices_)
    
    def set_ws(self,w_):
        pass
    
    def set_NL(self,NL_):
        pass
        
    def set_NL0(self,NL_):
        pass

    def set_NL1(self,NL_):
        pass

    def DF(self):
        self.df = np.exp(-self.r * self.time)        
        
    def zerorate(self, t):
        pass
  
    def ols(self, t):        
        pass
    
    def Zerorates(self):
        t      = self.time        
        vfunc = np.vectorize(self.zerorate)        
        self.r = vfunc(t)
    
    def loss(self, w):
        self.w = w
        self.Zerorates()
        self.DF()       
        Ax     = np.transpose(self.df).dot(self.Cashflows)
        error  = (self.Prices - Ax)**2                
        return np.mean(error)
    
    def Calibrate(self, initialguess):
        self.opt_results = optimize.minimize(self.loss,np.array(initialguess),jac=False,tol=1e-13,method='L-BFGS-B')
        self.w           = self.opt_results.x 
        
class NelsonSiegelSvensson(NelsonSiegelModels):
    def __init__(self, size):
        NelsonSiegelModels.__init__(self, size)
        self.NL  = np.zeros(2)
        self.w   = np.zeros(4)
    
    def set_ws(self,w_):
        self.w = w_
    
    def set_NL0(self,NL_):
        self.NL[0] = NL_

    def set_NL1(self,NL_):
        self.NL[1] = NL_
   
    def zerorate(self, t):
        vec = self.ols(t)        
        return self.w[0]*vec[0] + self.w[1] * vec[1] + self.w[2] * vec[2] + self.w[3] * vec[3]
    
    def ols(self, t):        
        a = 1.0
        b = ( (1-np.exp(-t/self.NL[0])) / (t/self.NL[0]))
        c = ( (1-np.exp(-t/self.NL[0])) / (t/self.NL[0]) - np.exp(-t/self.NL[0]) ) 
        d = ( (1-np.exp(-t/self.NL[1])) / (t/self.NL[1]) - np.exp(-t/self.NL[1]) )
        return np.asarray([a, b, c, d])

# and let's do the same for the Nelson Siegel algo
class NelsonSiegel(NelsonSiegelModels):
    def __init__(self, size):
        NelsonSiegelModels.__init__(self, size)
        self.NL  = np.zeros(1)
        self.w   = np.zeros(3)
            
    def set_ws(self, w_):
        self.w = w_
        
    def set_NL(self,NL_):
        self.NL = [NL_]
                
    def zerorate(self, t):        
        vec = self.ols(t)
        return self.w[0] * vec[0] + self.w[1] * vec[1] + self.w[2] * vec[2]
    
    def ols(self, t):
        a = 1.0
        b = ( (1-np.exp(-t/self.NL[0])) / (t/self.NL[0]) )
        c = ( (1-np.exp(-t/self.NL[0])) / (t/self.NL[0]) - np.exp(-t/self.NL[0]))        
        return np.asarray([a, b, c])