#cov_aff_inv_util.py
import numpy as np
import numpy.linalg as lin
import scipy as sc
import math

def dist(S, X):
    sqrm = sc.linalg.sqrtm(S)
    sqinv = lin.inv(sqrm)
    inner = sqinv @ X @ sqinv    
    logMat = sc.linalg.logm(inner)
    #eigs, P = lin.eigh(inner)
    #logMat = log_mat(eigs, P)
    return lin.norm(logMat)

def inner(A, B):
    return np.trace(A @ B)

# def inner(A, B, P):
#     Pinv = lin.inv(P)
#     prod = Pinv @ A @ Pinv @ B
#     return np.trace(prod)

# def to_diag(cov_mat_list):
#     if isinstance(cov_mat_list[0], tuple):
#         return cov_mat_list
#     return [lin.eigh(cm) for cm in cov_mat_list]


# def log_mat(evals, P, form='mat'):
#     if form == 'mat':
#         return P @ np.diag(np.log(evals)) @ lin.inv(P)
#     elif form == 'tup':
#         return np.log(evals), P
#     else:
#         ValueError("Unrecognized form: "+form+". Use tup or mat.")


def cov_mean(DataSeq):
    n = DataSeq[0].shape[0]
    K = len(DataSeq)

    pos = np.eye(n)

    tan = np.zeros((n, n))
    tan[0,1] = 1 
    tan[1,0] = 1

    step = 0.01 # Step size for gradient descent
    
    def Geo(Pos,Tan): 
        sqr = sc.linalg.sqrtm(Pos)
        return sqr*sc.linalg.expm(-step*Tan)*sqr

    def AffInvMetric(Pos,Tan):
        div = np.divide(Tan, Pos)
        return math.sqrt(np.trace(div @ div))

    def Grad(Pos):
        # Supporting Gradient function
        LogSeq = np.zeros((K,n,n))
        for k in range(K):
            LogSeq[k,:,:] = sc.linalg.logm(np.divide(Pos, DataSeq[k,:,:]))
        
        return Pos*np.sum(LogSeq, axis=0)

    # Intiate While Loop, Determine Minimum

    count = 0 # % Keep track of iterations

    while AffInvMetric(pos,tan) >= 0.1:
        pos   = Geo(pos,tan);
        tan   = Grad(pos);
        count += 1;

    Min = pos
    Iters = count

    return Min#, Iters

    

def cov_var(cov_mat_list):
    mu = cov_mean(cov_mat_list)

    norm = float(0)
    for S in cov_mat_list:
        norm += affinv_dist(S, mu)**2
    return norm/n


def cov_covar(X):
    m = X.shape[0]#number of rows
    n = X.shape[1]#number of sample covmats

    """assume that X is a list of vectors of sample covariances"""
    expectations = [cov_mean(x) for x in X]
    devs = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            devs[i,j] = dist(X[i,j], expectations[i])

    Sigma = (devs @ devs.T)/n
    return Sigma






