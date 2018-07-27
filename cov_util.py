#cov_log_eu_util.py
import numpy as np
import numpy.linalg as lin
import scipy.linalg as slin

def dist(S, X):
	#se, sp = lin.eigh(S)
	#xe, xp = lin.eigh(X)
	#logse = np.log(se.flatten())
	#logxe = np.log(xe.flatten())
	#logS = sp @ np.diag(logse) @ lin.inv(sp)
	#logX = xp @ np.diag(logxe) @ lin.inv(xp)
	logS = slin.logm(S)
	logX = slin.logm(X)
	return lin.norm(logS - logX)

def inner(A, B):
	return np.trace(A @ B)

# def dist2(se, sp, logX):
# 	logse = np.log(se)
# 	logS = sp @ np.diag(logse) @ lin.inv(sp)
# 	return lin.norm(logS - logX)


# def to_diag(cov_mat_list):
# 	if isinstance(cov_mat_list[0], tuple):
# 		return cov_mat_list
# 	return [lin.eigh(cm) for cm in cov_mat_list]

# def log_mat(evals, P, form='mat'):
# 	if form == 'mat':
# 		return P @ np.diag(np.log(evals)) @ lin.inv(P)
# 	elif form == 'tup':
# 		return np.log(evals), P
# 	else:
# 		ValueError("Unrecognized form: "+form+". Use tup or mat.")

def cov_mean(cov_mat_list):
	#cm_list = to_diag(cov_mat_list)
	sum_mat = np.zeros(cov_mat_list[0].shape)
	for cm in cov_mat_list:
		sum_mat += slin.logm(cm)#log_mat(cm[0], cm[1])
	sum_mat /= len(cov_mat_list)
	return slin.expm(sum_mat)
	#mean_evals, mean_P = lin.eigh(sum_mat)
	#return mean_P @ np.diag(np.exp(mean_evals)) @ lin.inv(mean_P)

def cov_var(cm_list):
	#cm_list = to_diag(cov_mat_list)
	mu = cov_mean(cm_list)
	#mu_D, mu_P = lin.eigh(mu)
	logmu = slin.logm(mu)#log_mat(mu_D, mu_P)
	n = len(cm_list)
	norm = float(0)
	for cm in cm_list:
		diff = slin.logm(cm) - logmu
		norm += lin.norm(diff)**2
	return norm/n


def cov_covar(X):
	m = X.shape.item(0)#number of rows
	n = X.shape.item(1)#number of sample covmats

	"""assume that X is a list of vectors of sample covariances, K x n x n"""
	expectations = [cov_mean(x) for x in X]
	devs = np.zeros((m,n))
	for i in range(m):
		for j in range(n):
			devs[i,j] = dist(X[i,j], expectations[i])

	Sigma = (devs @ devs.T)/n
	return Sigma






