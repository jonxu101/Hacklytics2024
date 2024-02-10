import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

col_names = ['Name', 'Risk', 'Return']

data = pd.read_csv('./riskreturn.csv')
n = len(data)
CovarRisk = np.diag(data['Risk'] ** 2)

MeanReturns = data['Return']
AEq = np.ones(MeanReturns.T.shape)

def MaximizeReturns(data, PortfolioSize): #optimize with no risk
    c = np.multiply(-1, data['Return'].tolist())
    A = np.ones([PortfolioSize, 1]).T
    b = [1]
    weights = opt.linprog(c, A_ub = A, b_ub = b, bounds = (0,1))
    return weights.x

def MinimizeReturns(data, size):
    def f(x):
        return np.matmul(np.matmul(x, CovarRisk), x.T)

    def constraintEq(x):
        return np.matmul(AEq,x.T)-1 
    
    xinit=np.repeat(0.1, size)
    cons = ({'type': 'eq', 'fun':constraintEq})
    lb = 0
    ub = 1
    bnds = tuple([(lb,ub) for x in xinit])

    weights = opt.minimize (f, x0 = xinit,  bounds = bnds, \
                             constraints = cons)
    
    return weights.x


def MeanVar(data, size, R):    
    def f(x):
        return np.matmul(np.matmul(x, CovarRisk), x.T) 

    def constraintEq(x):
        return np.matmul(AEq,x.T)-1 
    
    def constraintIneq(x, R):
        return np.matmul(np.array(MeanReturns), x.T) - R
    

    xinit=np.repeat(0.1, size)
    cons = ({'type': 'eq', 'fun':constraintEq},
            {'type':'ineq', 'fun':constraintIneq, 'args': [R] })
    lb = 0
    ub = 1
    bnds = tuple([(lb,ub) for x in xinit])

    res = opt.minimize (f, x0 = xinit, bounds = bnds, constraints = cons)
    return res.x

def ComputeReturns(data, weights):
    return np.matmul(np.array(data['Return'].tolist()), weights.T)

def ComputeRisk(data, weights):
    return np.sqrt(np.matmul(np.matmul(weights, CovarRisk), weights.T))

n = len(data)
incr = 0.1

min = ComputeReturns(data, MinimizeReturns(data, n))
max = ComputeReturns(data, MaximizeReturns(data, n))

print(min)
print(max)

frontier_risk = []
frontier_return = []

R = min # target return
while R < max:
    res = MeanVar(data, n, R)
    frontier_risk.append(ComputeRisk(data, res))
    frontier_return.append(R)
    R += incr
    # print(R)\
        
print("plotting")
plt.figure()
plt.scatter(frontier_risk, frontier_return)
plt.xlabel("Risk (%)")
plt.ylabel("Expected Return (%)")
plt.show()