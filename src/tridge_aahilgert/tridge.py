from sklearn.linear_model import Ridge, LogisticRegression
import numpy as np
import cupy as cp
from sys import maxsize
from scipy.optimize import minimize

get_xp = lambda x : cp.get_array_module(x)

def bFunction(x,theta,family):
    xp = get_xp(x)
    if family == "gaussian":
        return .5 * xp.square(x @ theta)
    elif family == "poisson":
        return xp.exp(x @ theta)
    else:
        return xp.log(1 + xp.exp(x @ theta))
    
def MeanFunction(x,theta,family):
    xp = get_xp(x)
    if family == "gaussian":
        return x @ theta
    elif family == "poisson":
        return xp.exp(x @ theta)
    else:
        return xp.exp(x @ theta) / (1 + xp.exp(x @ theta))
    
def MeanPrime(x,theta,family):
    xp = get_xp(x)
    if family == "gaussian":
        return xp.ones(x.shape[1]) 
    elif family == "poisson":
        return xp.exp(x @ theta)
    else:
        return xp.exp(x @ theta) / xp.square(1+xp.exp(x @ theta))
    
def ObjectiveFunction(theta,family,y,x,trex_c):
    xp = get_xp(x)
    
    norm = lambda a : xp.linalg.norm(a,ord=2)
    sq = lambda a : a**2
    
    if family == "gaussian":
        loss = y - x @ theta
        derivative = x.T @ (y - x @ theta)
        return (sq(norm(loss))- sq(norm(y))) / (trex_c * norm(derivative)) + norm(theta)

    else:
        loss = xp.sum(y * (x @ theta) - bFunction(x, theta,family))
        tune_vector = ((y - MeanFunction(x, theta,family)).T @ x)
        return -loss / (trex_c * norm(tune_vector)) + norm(theta)
    
def GradientLs(theta,x,y,family):
    xp = get_xp(x)
    
    if family=="gaussian":
        return -2 * x.T @ (y - x @ theta)
    else:
        return -((y - MeanFunction(x,theta,family)).T @ x).T # not t in led
    
class ObLs:

    def __init__(self, x, y, family):
        self.xp = get_xp(x)
        self.x = x
        self.y = y
        self.family = family

    def operator(s, theta):
        
        norm = lambda a : s.xp.linalg.norm(a,ord=2)
        sq = lambda a : a**2
        
        if s.family=="gaussian": 
            loss = s.y - s.x @ theta;
            return sq(norm(loss)) - sq(norm(s.y))

        else:
            return -s.xp.sum(s.y*(s.x@theta) - bFunction(s.x,theta,s.family))
          
 
    def gradient(s, theta):
        
        norm = lambda a : s.xp.linalg.norm(a,ord=2)
        sq = lambda a : a**2
        
        if s.family=="gaussian":
            derivative = s.x.T @ (s.y - s.x @ theta)
            return -2 * derivative
        else:
            tune_vector = ((s.y - MeanFunction(s.x,theta,s.family)).T @ s.x).T
            return -tune_vector
        
def Optim_Obls(theta,x,y,family):
    func = ObLs(x,y,family)
    minx = minimize(func.operator,theta,jac=func.gradient,method="CG", options={'maxiter': 10000})
    return minx.x

class ObRidge:

    def __init__(self, x, y, family, lam):
        self.xp = get_xp(x)
        self.x = x
        self.y = y
        self.family = family
        self.lam = lam
    

    def operator(s, theta):
        
        norm = lambda a : s.xp.linalg.norm(a,ord=2)
        sq = lambda a : a**2
        
        if s.family=="gaussian": 
            loss = s.y - s.x @ theta;
            return sq(norm(loss))- sq(norm(s.y)) + s.lam*sq(norm(theta))
        else:
            loss = s.xp.sum(s.y * (s.x @ theta) - bFunction(s.x,theta,s.family))
            return -loss+s.lam*sq(norm(theta))
          
 
    def gradient(s, theta):
        
        norm = lambda a : s.xp.linalg.norm(a,ord=2)
        sq = lambda a : a**2
        
        if s.family=="gaussian":
            derivative = s.x.T @ (s.y - s.x @ theta)
            return -2 * derivative + s.lam * 2 *norm(theta)
        else:
            tune_vector = ((s.y - MeanFunction(s.x,theta,s.family)).T @ s.x).T
            return -tune_vector + s.lam * 2 * norm(theta)
        
        
def optim_Ridge(theta,x,y,family,lam):
    func = ObRidge(x,y,family,lam)
    minx = minimize(func.operator,theta,jac=func.gradient,method="CG", options={'maxiter': 10000})
    return minx.x


class ObFn:

    def __init__(self, x, y, family, trex_c):
        self.xp = get_xp(x)
        self.x = x
        self.y = y
        self.family = family
        self.trex_c = trex_c
    
    

    def operator(s, theta):
        
        norm = lambda a : s.xp.linalg.norm(a,ord=2)
        sq = lambda a : a**2
        
        if s.family=="gaussian": 
            
            loss = s.y - s.x@theta;
            derivative = s.x.T @ (s.y - s.x@theta)
            return ((sq(norm(loss))- sq(norm(s.y))) / (s.trex_c * norm(derivative)) + norm(theta))

        else:
            loss = s.xp.sum(s.y * (s.x @ theta) - bFunction(s.x, theta,s.family))
            tune_vector = ((s.y - MeanFunction(s.x, theta,s.family)).T @ s.x).T
            regularization = norm(theta)
            return -loss /  (s.trex_c * norm(tune_vector)) + regularization
          
 
    def gradient(s, theta):
        
        norm = lambda a : s.xp.linalg.norm(a,ord=2)
        sq = lambda a : a**2
        cube = lambda a : a**3
        
        if s.family=="gaussian":
            
            derivative = s.x.T @ (s.y - s.x @ theta)
            loss = s.y - s.x @ theta
            return (
                (sq(norm(loss))- sq(norm(s.y))) * s.x.T @ s.x @ derivative 
                / (s.trex_c * cube(norm(derivative))) 
                - 2 * derivative / (s.trex_c * norm(derivative)) +
                theta /  norm(theta)
            )
        else:
            meanprime = MeanPrime(s.x,theta,s.family)
            loss = s.xp.sum(s.y * (s.x @ theta) - bFunction(s.x,theta,s.family))
            tune_vector = ((s.y - MeanFunction(s.x,theta,s.family)).T @ s.x).T
            
            return (
                - tune_vector / (s.trex_c * norm(tune_vector)) 
                - (s.trex_c * loss * tune_vector.T @ s.x.T @ s.xp.apply_along_axis(lambda a:a*meanprime,0,s.x) 
                / (sq(s.trex_c) * cube(norm(tune_vector)))).T 
                + theta / (1 if norm(theta) == 0 else norm(theta))
            )
        
def optim_ObFn(theta,x,y,family,trex_c):
    func = ObFn(x,y,family,trex_c)
    minx = minimize(func.operator,theta,jac=func.gradient,method="CG", options={'maxiter': 10000})
    return minx.x

def pit(xp, x, Omega, piter = 3):
    y = x @ Omega
    for q in range(piter):
        y = x @ (x.T @ y)
    q, _ = xp.linalg.qr(y)
    return q

def rsvd(xp, A):
    Omega = xp.random.randn(A.shape[1], min(A.shape[0],A.shape[1]))
    Q = pit(xp, A, Omega)
    B = Q.T @ A
    u_tilde, s, v = xp.linalg.svd(B, full_matrices=False)
    u = Q @ u_tilde
    return u, s, v

def glmnet_binomial(lam,beta_old,X,y):
    
    xp = get_xp(X)
    
    inv = lambda a : xp.linalg.inv(a) 
    
    threshhold = 1000

    INN=xp.identity(X.shape[0])
    
    beta_new = beta_old
  
    i = 0

    while threshhold>0.1*y.shape[0] and i<50:
    
        diagonal_w=xp.exp(X@beta_old)/xp.square(1+xp.exp(X@beta_old))
        
        W=xp.diag(diagonal_w.reshape((X.shape[0],)))
                  
        diagonal_w_inv=xp.square(1+xp.exp(X@beta_old))/xp.exp(X@beta_old)
     
        W_inv=xp.diag(diagonal_w_inv.reshape((X.shape[0],)))
    
        g_inverse=xp.exp(X@beta_old)/(1+xp.exp(X@beta_old))
        
        Z= X@beta_old + W_inv @(y-g_inverse)
        
        beta_new=(1/lam)*X.T@W@Z-(1/(lam**2))*X.T@inv(INN+(1/lam)*W@X@X.T)@W@X@X.T@W@Z
    
        threshhold = xp.linalg.norm(beta_old-beta_new,ord=1)
    
        beta_old=beta_new
        i += 1
  
    return beta_old

def glmnet_poisson(lam,beta_old,X,y):
    
    xp = get_xp(X)
    
    inv = lambda a : xp.linalg.inv(a) 

    threshhold = 1000

    INN=xp.identity(X.shape[0])
    
    beta_new = beta_old
  
    i = 0
  
    while threshhold>0.1*y.shape[0] and i<50:
    
        diagonal_w = xp.exp(X@beta_old)
    
        W = xp.diag(diagonal_w.reshape((X.shape[0],)))
        
        diagonal_w_inv = 1/xp.exp(X@beta_old)
        
        W_inv = xp.diag(diagonal_w_inv.reshape((X.shape[0],)))
        
        g_inverse=xp.exp(X@beta_old)
        
        Z= X@beta_old + W_inv @(y-g_inverse)
        
        beta_new=(1/lam)*X.T@W@Z-(1/(lam**2))*X.T@inv(INN+(1/lam)*W@X@X.T)@W@X@X.T@W@Z
    
        threshhold = xp.linalg.norm(beta_old-beta_new,ord=1)
    
        beta_old=beta_new
        i += 1
        
    return beta_old


def tridge(x,y,family,r_type=None,nlambda=100,lambda_min=0.05,c=5,nfolds=10,alpha=0,stn=10):
    
    xp = get_xp(x)
    
    norm = lambda a : xp.linalg.norm(a,ord=2)
    
    trex_vec = 2 if family == "gaussian" else 1
    
    observations, params = x.shape
    
    param_0 = xp.zeros(params) 
    param_1 = Optim_Obls(param_0,x,y,family)
    
    GlmTrex_estimators = optim_ObFn(param_1,x,y,family,trex_vec)

    edr_lambda = norm(GradientLs(GlmTrex_estimators,x,y,family)) / (2 * norm(GlmTrex_estimators))
    
    r_max = edr_lambda + c
    r_min = max(lambda_min, (edr_lambda - c))
    
    if r_min >= maxsize or r_max >= maxsize or np.isnan(r_min) or np.isnan(r_max):
        tuning_parameters = xp.linspace(1e+10, 1e+11, nlambda)
    else:
        tuning_parameters = xp.linspace(r_min, r_max, nlambda) 
        
        
    if family=="gaussian":
        if r_type=="rsvd":

            u, d, v = rsvd(xp, x)
            d = xp.diag(d)


        elif r_type=="svd":

            u, d, v = xp.linalg.svd(x, full_matrices=False)
            d = xp.diag(d)
        
    
    inv = lambda a : xp.linalg.inv(a) 

    IN = xp.identity(x.shape[0])
 
    if family=="gaussian":
        if r_type in ("rsvd","svd"):
            D_2 = d@d
            R = u@d
            RTR = R.T@R
            RTY = R.T@y
        else:
            XXT = x@x.T
            XTY = x.T@y
            XXTY = XXT@y
            
    elif family == "binomial":
        cv_chosen_b = glmnet_binomial(edr_lambda,param_1,x,y)
    else:
        cv_chosen_b = glmnet_poisson(edr_lambda,param_1,x,y)
        
    
    if family == "gaussian":
            if r_type in ("rsvd","svd"):
                #functor = lambda a : v.T@inv(RTR + a*IN)@RTY
                functor = lambda a : v.T@inv(D_2 + a*IN)@RTY

            else:
                functor = lambda a : XTY/a - (x.T@(inv(IN + XXT/a) @ XXTY)) / (a**2)
    
    elif family == "binomial":
            
        functor = lambda a : glmnet_binomial(a,cv_chosen_b,x,y)
        
    else:
        
        functor = lambda a : glmnet_poisson(a,cv_chosen_b,x,y)

    
    cost_functor = lambda a : ObjectiveFunction(a,family,y,x,trex_vec)
    
    estimators = xp.apply_along_axis(functor,1,tuning_parameters.reshape(nlambda,1))
    
    #estimators = estimators[:, ~np.isnan(estimators).any(axis=0)]
    
    costs = xp.apply_along_axis(cost_functor,1,estimators)
    
    return estimators[xp.argmin(costs)]

def sklearn_ridge_gaussian(lamb, x, y):
    est = Ridge(alpha=float(lamb))
    est.fit(x, y)
    return est.coef_.reshape(x.shape[1],)

def sklearn_ridge_binomial(lamb, x, y):
    est = LogisticRegression(C=1/float(lamb))
    est.fit(x, y)
    return est.coef_.reshape(x.shape[1],)


class GaussianRegressor:
    
    def __init__(self,nlambda=100,lambda_min=0.05,c=5,nfolds=10,processor=None,r_type=None,function=None):
        
        self.family = "gaussian"
        self.nlambda = nlambda
        self.lambda_min = lambda_min
        self.c = c
        self.nfolds = nfolds
        self.lambdas = None
        
        if processor not in ["gpu","cpu"] or (processor == "gpu" and r_type not in [None, "woodbury","svd","rsvd","custom"]):
            self.processor = "cpu"

        else:
            self.processor = processor
       
        self.xp = np if self.processor == "cpu" else cp
        
        if (
            r_type == None 
            or r_type not in ["woodbury","svd","rsvd","sklearn","custom"] 
            or (r_type == "custom" and function is None)
        ):
            self.r_type = "woodbury"
        elif r_type == "custom":
            self.r_type = r_type
            self.function == function
        else:
            self.r_type = r_type
            
    def gen_lambdas(self, x, y):
        
        if self.xp == cp:
            if get_xp(x) == np: 
                self.x = cp.asarray(x)
            else:
                self.x = x
            if get_xp(y) == np:
                self.y = cp.asarray(y)
            else:
                self.y = y
        else:
            if get_xp(x) == cp: 
                self.x = cp.asnumpy(x.get())
            else:
                self.x = x
            if get_xp(y) == cp:
                self.y = cp.asnumpy(y.get())
            else:
                self.y = y
        
        if get_xp(x) == cp: 
            x = cp.asnumpy(x.get())
        if get_xp(y) == cp:
            y = cp.asnumpy(y.get())
    
        norm = lambda a : np.linalg.norm(a,ord=2)

        self.trex_vec = 2 

        observations, params = x.shape

        param_0 = np.zeros(params) 
        param_1 = Optim_Obls(param_0,x,y,self.family)

        GlmTrex_estimators = optim_ObFn(param_1,x,y,self.family,self.trex_vec)

        edr_lambda = norm(GradientLs(GlmTrex_estimators,x,y,self.family)) / (2 * norm(GlmTrex_estimators))

        r_max = edr_lambda + self.c
        r_min = max(self.lambda_min, (edr_lambda - self.c))

        if r_min >= maxsize or r_max >= maxsize or np.isnan(r_min) or np.isnan(r_max):
            tuning_parameters = self.xp.linspace(1e+10, 1e+11, self.nlambda)
        else:
            tuning_parameters = self.xp.linspace(r_min, r_max, self.nlambda) 
            
        self.lambdas = tuning_parameters.reshape(self.nlambda,1)
        
    def fit(self, x, y):
        
        self.gen_lambdas(x, y)
            
        if self.r_type in ("woodbury","rsvd","svd"):
            inv = lambda a : self.xp.linalg.inv(a) 
            IN = self.xp.identity(self.x.shape[0])

            if self.r_type in ("rsvd","svd"):
                if self.r_type=="rsvd":
                    u, d, v = rsvd(self.xp, self.x)
                    d = self.xp.diag(d)
                else:
                    u, d, v = self.xp.linalg.svd(self.x, full_matrices=False)
                    d = self.xp.diag(d)
                D_2 = d@d
                R = u@d
                RTR = R.T@R
                RTY = R.T@y
                function = lambda a : v.T@inv(D_2 + a*IN)@RTY
            else:
                XXT = self.x@self.x.T
                XTY = self.x.T@y
                XXTY = XXT@y
                function = lambda a : XTY/a - (self.x.T@(inv(IN + XXT/a) @ XXTY)) / (a**2)
                
        elif self.r_type == "sklearn":
            function = lambda a : sklearn_ridge_gaussian(a, self.x, self.y)
        else:
            function = self.function
        
        cost_functor = lambda a : ObjectiveFunction(a,self.family,self.y,self.x,self.trex_vec)

        estimators = self.xp.apply_along_axis(function,1,self.lambdas)

        costs = self.xp.apply_along_axis(cost_functor,1,estimators)

        self.best_lambda = self.lambdas[self.xp.argmin(costs)]
        self.best_cost = self.xp.min(costs)
        self.coef_ = estimators[self.xp.argmin(costs)]
        
        
class BinomialRegressor:
    
    def __init__(self,nlambda=100,lambda_min=0.05,c=5,nfolds=10,processor=None,r_type=None,function=None):
        
        self.family = "binomial"
        self.nlambda = nlambda
        self.lambda_min = lambda_min
        self.c = c
        self.nfolds = nfolds
        
        if processor not in ["gpu","cpu"] or (processor == "gpu" and r_type not in ["rls","custom"]):
            self.processor = "cpu"
        else:
            self.processor = processor
            
        self.xp = np if self.processor == "cpu" else cp
        
        if (
            r_type == None 
            or r_type not in ["sklearn","rls","custom"] 
            or (r_type == "custom" and function is None)
        ):
            self.r_type = "sklearn"
        elif r_type == "custom":
            self.r_type = r_type
            self.function = function
        else:
            self.r_type = r_type
            
    def gen_lambdas(self, x, y):
        
        if self.xp == cp:
            if get_xp(x) == np: 
                self.x = cp.asarray(x)
            else:
                self.x = x
            if get_xp(y) == np:
                self.y = cp.asarray(y)
            else:
                self.y = y
        else:
            if get_xp(x) == cp: 
                self.x = cp.asnumpy(x.get())
            else:
                self.x = x
            if get_xp(y) == cp:
                self.y = cp.asnumpy(y.get())
            else:
                self.y = y
        
        if get_xp(x) == cp: 
            x = cp.asnumpy(x.get())
        if get_xp(y) == cp:
            y = cp.asnumpy(y.get())
    
        norm = lambda a : np.linalg.norm(a,ord=2)

        self.trex_vec = 1 

        observations, params = x.shape

        param_0 = np.zeros(params) 
        param_1 = Optim_Obls(param_0,x,y,self.family)
        
        self.cv_chosen_b = param_1 if self.xp == np else cp.asarray(param_1)

        GlmTrex_estimators = optim_ObFn(param_1,x,y,self.family,self.trex_vec)

        edr_lambda = norm(GradientLs(GlmTrex_estimators,x,y,self.family)) / (2 * norm(GlmTrex_estimators))

        r_max = edr_lambda + self.c
        r_min = max(self.lambda_min, (edr_lambda - self.c))

        if r_min >= maxsize or r_max >= maxsize or np.isnan(r_min) or np.isnan(r_max):
            tuning_parameters = self.xp.linspace(1e+10, 1e+11, self.nlambda)
        else:
            tuning_parameters = self.xp.linspace(r_min, r_max, self.nlambda) 
            
        self.lambdas = tuning_parameters.reshape(self.nlambda,1)
        
    def fit(self, x, y):

        self.gen_lambdas(x, y)

        if self.r_type == "sklearn":
            function = lambda a : sklearn_ridge_binomial(a, self.x, self.y)
        elif self.r_type == "rls":
            function = lambda a : glmnet_binomial(a,self.cv_chosen_b,self.x,self.y)
        else:
            function = self.function

        cost_functor = lambda a : ObjectiveFunction(a,self.family,self.y,self.x,self.trex_vec)

        estimators = self.xp.apply_along_axis(function,1,self.lambdas)

        costs = self.xp.apply_along_axis(cost_functor,1,estimators)

        self.best_lambda = self.lambdas[self.xp.argmin(costs)]
        self.best_cost = self.xp.min(costs)
        self.coef_ = estimators[self.xp.argmin(costs)]
        
        
class PoissonRegressor:
    
    def __init__(self,nlambda=100,lambda_min=0.05,c=5,nfolds=10,processor=None,r_type=None,function=None):
        
        self.family = "poisson"
        self.nlambda = nlambda
        self.lambda_min = lambda_min
        self.c = c
        self.nfolds = nfolds
        
        if processor not in ["gpu","cpu"]:
            self.processor = "cpu"
        else:
            self.processor = processor
            
        self.xp = np if self.processor == "cpu" else cp
        
        if (
            r_type == None 
            or r_type not in ["rls","custom"] 
            or (r_type == "custom" and function is None)
        ):
            self.r_type = "rls"
        elif r_type == "custom":
            self.r_type = r_type
            self.function == function
        else:
            self.r_type = r_type
            
    def gen_lambdas(self, x, y):
        
        if self.xp == cp:
            if get_xp(x) == np: 
                self.x = cp.asarray(x)
            else:
                self.x = x
            if get_xp(y) == np:
                self.y = cp.asarray(y)
            else:
                self.y = y
        else:
            if get_xp(x) == cp: 
                self.x = cp.asnumpy(x.get())
            else:
                self.x = x
            if get_xp(y) == cp:
                self.y = cp.asnumpy(y.get())
            else:
                self.y = y
        
        if get_xp(x) == cp: 
            x = cp.asnumpy(x.get())
        if get_xp(y) == cp:
            y = cp.asnumpy(y.get())
    
        norm = lambda a : np.linalg.norm(a,ord=2)

        self.trex_vec = 1 

        observations, params = x.shape

        param_0 = np.zeros(params) 
        param_1 = Optim_Obls(param_0,x,y,self.family)
        self.cv_chosen_b = param_1 if self.xp == np else cp.asarray(param_1)

        GlmTrex_estimators = optim_ObFn(param_1,x,y,self.family,self.trex_vec)

        edr_lambda = norm(GradientLs(GlmTrex_estimators,x,y,self.family)) / (2 * norm(GlmTrex_estimators))
        self.edr_lambda = edr_lambda

        r_max = edr_lambda + self.c
        r_min = max(self.lambda_min, (edr_lambda - self.c))

        if r_min >= maxsize or r_max >= maxsize or np.isnan(r_min) or np.isnan(r_max):
            tuning_parameters = self.xp.linspace(1e+10, 1e+11, self.nlambda)
        else:
            tuning_parameters = self.xp.linspace(r_min, r_max, self.nlambda) 
            
        self.lambdas = tuning_parameters.reshape(self.nlambda,1)
        
    def fit(self, x, y):
        
        self.gen_lambdas(x, y)

        if self.r_type == "rls":
            function = lambda a : glmnet_poisson(a,self.cv_chosen_b,self.x,self.y)
        else:
            function = self.function

        cost_functor = lambda a : ObjectiveFunction(a,self.family,self.y,self.x,self.trex_vec)

        estimators = self.xp.apply_along_axis(function,1,self.lambdas)

        costs = self.xp.apply_along_axis(cost_functor,1,estimators)

        self.best_lambda = self.lambdas[self.xp.argmin(costs)]
        self.best_cost = self.xp.min(costs)
        self.coef_ = estimators[self.xp.argmin(costs)]