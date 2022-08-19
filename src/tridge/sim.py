import time
import numpy as np
from tridge import GaussianRegressor, BinomialRegressor, PoissonRegressor
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from glmnet import LogitNet, ElasticNet

def data_gen(case, obs, par, k, SNR=10, class_n=1):
    
    mu = np.zeros(par)
    sigma = np.ndarray((par,par))
    for i in range(par):
        for j in range(par):
            sigma[i,j] = k ** abs(i-j)
            
    X = np.random.multivariate_normal(mu, sigma, obs)
    
    X = X / np.linalg.norm(X, 2, axis=0)
    
    if obs < par:
          _, _, v = np.linalg.svd(X, full_matrices=False)
    else:
          _, _, v = np.linalg.svd(X, full_matrices=True)
            
    proj = v.T @ v

    beta_p = np.random.normal(0, 1, par) 
    
    beta = proj @ beta_p

    if case == "gaussian":
        y = np.random.normal(X @ beta, np.sqrt((1/SNR) * (beta@sigma@beta)))
    elif case == "poisson":
        y = np.random.poisson(np.exp(X @ beta)/np.sqrt(SNR))
    else:
        y = np.random.binomial(class_n, (1 / (1 + np.exp(-X @ beta)))/np.sqrt(SNR))
        
    return X, beta, y, sigma, SNR
    
    
def sim_gaussian(obs, par, k, SNR):
    
    data_aglo = []
    
    X, beta, y, sigma, SNR = data_gen("gaussian", obs, par, k, SNR)
    for processor in ['gpu','cpu']:
        for r_type in ['woodbury','svd','rsvd']:
    
            reg = GaussianRegressor(processor=processor,r_type=r_type)
            reg.fit(X,y)
            analysis_dict = reg.full_analysis(beta, sigma, SNR)
            data_aglo.append(analysis_dict)
    
    
    glmnet_reg = ElasticNet(alpha=0, n_splits=5, fit_intercept=False, n_lambda=100)
    t0 = time.time()
    glmnet_reg.fit(X,y)
    t1 = time.time()
    reg.coef_ = glmnet_reg.coef_
    analysis_dict = reg.full_analysis(beta, sigma, SNR)
    analysis_dict['algorithm'] = "glmnet_cv5"
    analysis_dict['time'] = t1-t0
    data_aglo.append(analysis_dict)
    
    glmnet_reg = ElasticNet(alpha=0, n_splits=10, fit_intercept=False, n_lambda=100)
    t0 = time.time()
    glmnet_reg.fit(X,y)
    t1 = time.time()
    reg.coef_ = glmnet_reg.coef_
    analysis_dict = reg.full_analysis(beta, sigma, SNR)
    analysis_dict['algorithm'] = "glmnet_cv10"
    analysis_dict['time'] = t1-t0
    data_aglo.append(analysis_dict)
    
    sklearn_reg = RidgeCV(fit_intercept=False,cv=5)
    t0 = time.time()
    sklearn_reg.fit(X,y)
    t1 = time.time()
    reg.coef_ = sklearn_reg.coef_
    analysis_dict = reg.full_analysis(beta, sigma, SNR)
    analysis_dict['algorithm'] = "sklearn_cv5"
    analysis_dict['time'] = t1-t0
    data_aglo.append(analysis_dict)
    
    sklearn_reg = RidgeCV(fit_intercept=False,cv=10)
    t0 = time.time()
    sklearn_reg.fit(X,y)
    t1 = time.time()
    reg.coef_ = sklearn_reg.coef_
    analysis_dict = reg.full_analysis(beta, sigma, SNR)
    analysis_dict['algorithm'] = "sklearn_cv10"
    analysis_dict['time'] = t1-t0
    data_aglo.append(analysis_dict)
    
    return data_aglo
    
    
def sim_binomial(obs, par, k, SNR):
    
    data_aglo = []
    
    X, beta, y, sigma, SNR = data_gen("binomial", obs, par, k, SNR)
    for processor in ['gpu','cpu']:
        reg = BinomialRegressor(processor=processor,r_type='rls')
        reg.fit(X,y)
        analysis_dict = reg.full_analysis(beta, sigma, SNR) 
        data_aglo.append(analysis_dict)
        
    glmnet_reg = LogitNet(alpha=0, n_splits=5, fit_intercept=False, n_lambda=100)
    t0 = time.time()
    glmnet_reg.fit(X,y)
    t1 = time.time()
    reg.coef_ = glmnet_reg.coef_.reshape((X.shape[1],))
    analysis_dict = reg.full_analysis(beta, sigma, SNR)
    analysis_dict['algorithm'] = "glmnet_cv5"
    analysis_dict['time'] = t1-t0
    data_aglo.append(analysis_dict)
    
    glmnet_reg = LogitNet(alpha=0, n_splits=10, fit_intercept=False, n_lambda=100)
    t0 = time.time()
    glmnet_reg.fit(X,y)
    t1 = time.time()
    reg.coef_ = glmnet_reg.coef_.reshape((X.shape[1],))
    analysis_dict = reg.full_analysis(beta, sigma, SNR)
    analysis_dict['algorithm'] = "glmnet_cv10"
    analysis_dict['time'] = t1-t0
    data_aglo.append(analysis_dict)
    
    sklearn_reg = LogisticRegressionCV(Cs=100,fit_intercept=False,cv=5)
    t0 = time.time()
    sklearn_reg.fit(X,y)
    t1 = time.time()
    reg.coef_ = sklearn_reg.coef_.reshape((X.shape[1],))
    analysis_dict = reg.full_analysis(beta, sigma, SNR)
    analysis_dict['algorithm'] = "sklearn_cv5"
    analysis_dict['time'] = t1-t0
    data_aglo.append(analysis_dict)
    
    sklearn_reg = LogisticRegressionCV(Cs=100,fit_intercept=False,cv=10)
    t0 = time.time()
    sklearn_reg.fit(X,y)
    t1 = time.time()
    reg.coef_ = sklearn_reg.coef_.reshape((X.shape[1],))
    analysis_dict = reg.full_analysis(beta, sigma, SNR)
    analysis_dict['algorithm'] = "sklearn_cv10"
    analysis_dict['time'] = t1-t0
    data_aglo.append(analysis_dict)
    
    return data_aglo
    
    
    
def sim_poisson(obs, par, k, SNR):
    
    data_aglo = []
    
    X, beta, y, sigma, SNR = data_gen("poisson", obs, par, k, SNR)
    for processor in ['gpu','cpu']:
        reg = PoissonRegressor(processor=processor)
        reg.fit(X,y)
        analysis_dict = reg.full_analysis(beta, sigma, SNR)
        data_aglo.append(analysis_dict)
        
    glmnet_reg = ElasticNet(alpha=0, n_splits=5, fit_intercept=False, n_lambda=100)
    t0 = time.time()
    glmnet_reg.fit(X,y)
    t1 = time.time()
    reg.coef_ = glmnet_reg.coef_
    analysis_dict = reg.full_analysis(beta, sigma, SNR)
    analysis_dict['algorithm'] = "glmnet_cv5"
    analysis_dict['time'] = t1-t0
    data_aglo.append(analysis_dict)
    
    glmnet_reg = ElasticNet(alpha=0, n_splits=10, fit_intercept=False, n_lambda=100)
    t0 = time.time()
    glmnet_reg.fit(X,y)
    t1 = time.time()
    reg.coef_ = glmnet_reg.coef_
    analysis_dict = reg.full_analysis(beta, sigma, SNR)
    analysis_dict['algorithm'] = "glmnet_cv10"
    analysis_dict['time'] = t1-t0
    data_aglo.append(analysis_dict)
    
    sklearn_reg = RidgeCV(fit_intercept=False,cv=5)
    t0 = time.time()
    sklearn_reg.fit(X,y)
    t1 = time.time()
    reg.coef_ = sklearn_reg.coef_
    analysis_dict = reg.full_analysis(beta, sigma, SNR)
    analysis_dict['algorithm'] = "sklearn_cv5"
    analysis_dict['time'] = t1-t0
    data_aglo.append(analysis_dict)
    
    sklearn_reg = RidgeCV(fit_intercept=False,cv=10)
    t0 = time.time()
    sklearn_reg.fit(X,y)
    t1 = time.time()
    reg.coef_ = sklearn_reg.coef_
    analysis_dict = reg.full_analysis(beta, sigma, SNR)
    analysis_dict['algorithm'] = "sklearn_cv10"
    analysis_dict['time'] = t1-t0
    data_aglo.append(analysis_dict)
    
    return data_aglo
    
def simulation(entry):
    
    s_entry = entry.split(',')
    
    case = s_entry[0]
    obs = int(s_entry[1])
    par = int(s_entry[2])
    k = float(s_entry[3])
    SNR = float(s_entry[4])
    
    
    if case == "gaussian":
        l_data = sim_gaussian(obs, par, k, SNR)
    elif case == "binomial":
        l_data = sim_binomial(obs, par, k, SNR)
    else:
        l_data = sim_poisson(obs, par, k, SNR)
        
    joined_l_data = []
    for d_entry in l_data:
        joined_l_data.append(','.join(str(x) for x in d_entry.values()))
        
    return '\n'.join(joined_l_data)


def file_sim(in_file,out_file):
    
    with open(in_file,'r') as f:
        contents = f.read()
        
    entry_list = contents.split('\n')[:-1]

    header = 'family,algorithm,n,p,relative risk,relative test error gaussian,proportion of variance explained gaussian,relative test error,proportion of variance explained,beta error,relative beta error,relative prediction error,corr,time'

    with open(out_file,'a') as f:
        f.write(header+'\n')

    for i, line in enumerate(entry_list):
        result = simulation(line)
        with open(out_file, 'a') as f:
            f.write(result+'\n')
        print("entry",i,"finished")
        

def gen_schedule(in_file):

    entry_list = []

    for case in ['gaussian','binomial','poisson']:
        for obs in np.linspace(20,100000,num=20):
            for par in np.linspace(20,100000,num=20):
                for k in [0.2,0.4,0.6,0.8,1]:
                    for SNR in [0.1,0.5,1,5,10,20]:
                        entry_list.append(','.join([case,str(int(obs)),str(int(par)),str(k),str(SNR)]))

    entry = '\n'.join(entry_list)

    with open(in_file,'w') as f:
        f.write(entry+'\n')


