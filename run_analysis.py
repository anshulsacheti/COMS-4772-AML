import sys
import numpy as np
from scipy.stats import threshold
from numpy.linalg import inv
from pandas import *
from parse_pyvcf import get_matrices
from pprint import pprint

def run_softimpute(M):
    tol = 1e-10
    n,p=M.shape
    X = DataFrame(np.zeros((n,p)))
    M = DataFrame(M)
    for i in np.arange(1.0,0.01,-0.01):
        sftdiff = np.where(M.isnull(), X, M)

        U,S,V = np.linalg.svd(sftdiff,full_matrices=False)
        S = np.diag(S)
        Slmb = np.maximum(S-i,np.zeros(S.shape))
        X_new = np.dot(U, np.dot(S, V))
        diff = np.linalg.norm(X_new-X,'fro')**2/np.linalg.norm(X_new,'fro')**2 
        print "i:", i, "diff:", diff
        if diff < tol:
            return X_new
        X = X_new

def run_lmm_alternate(X,Y,U,G):
    
    Ulmm = X
    Umeans = Ulmm.mean(axis=0)
    Ulmm = Ulmm - Umeans
    Glmm = np.cov(Ulmm.transpose())
    Xlmm = U
    Xmeans = Xlmm.mean(axis=0)
    Xlmm = Xlmm - Xmeans
    Y = Y.values[:,[0]] # eye color
    Y = Y - Y.mean(axis=0)
    rho = 0.1
    V = np.dot(Ulmm, np.dot(Glmm,Ulmm.transpose())) + rho*np.cov(Ulmm)
 

    beta_inv = np.dot(Xlmm.transpose(),np.dot(np.linalg.inv(V), Xlmm))
    beta = np.linalg.inv(beta_inv)
    beta = np.dot(beta, Xlmm.transpose())
    beta = np.dot(beta, np.dot(np.linalg.inv(V),Y))
    pprint(beta)

    gamma = Glmm.dot(Ulmm.transpose()).dot(np.linalg.inv(V)).dot(Y-Xlmm.dot(beta))
    pprint(gamma)

    print "Ulmm:", np.max(Ulmm), np.min(Ulmm)
    print "Glmm:", np.max(Glmm), np.min(Glmm)
    print "Xlmm:", np.max(Xlmm), np.min(Xlmm)


def soft_thresh(M,tau):
    U,S,V = np.linalg.svd(M,full_matrices=False)
    S = np.diag(S)
    Stau = np.maximum(S-tau,np.zeros(S.shape))
    M_new = U.dot(Stau).dot(V)
    return M_new

def run_admm(X,Y,U,G):
    #X = X.as_matrix()
    #Y = Y.as_matrix()
    Y = Y.ix[:,['Height']]
    Xp = DataFrame(X).fillna(0).as_matrix()
    Xpp = DataFrame(X).fillna(0).as_matrix()
    u = np.zeros(X.shape)
    beta_ols = np.zeros((X.shape[1],1))
    beta_lasso = np.zeros((X.shape[1],1))

    lambda1 = 0.1
    lambda2 = 0.1
    lambda3 = 0.1
    rho = 0.1

    # place holder
    V = np.identity(X.shape[0])

    tol = 0.001
    diff = 999
    i = 0
    while i < 10:
        X_Xp_sub = np.where(X.isnull(), Xp, X)
        Xp = soft_thresh(1/(lambda2+rho) * (lambda2*X_Xp_sub + u + rho*Xpp), lambda3/(lambda2+rho))

        print "got here 1"

        a = (rho*Xp + Y.dot(beta_lasso.transpose()) - u)
        pprint(a)
        print a.shape
        b = inv(beta_lasso.dot(beta_lasso.transpose())+rho*np.identity(beta_lasso.shape[0]))
        pprint(b)
        print b.shape
        Xpp = a.dot(b)
        #Xpp = (rho*Xp + Y.dot(beta_lasso.transpose()) - u).dot(inv(beta_lasso.dot(beta_lasso.transpose())+rho*np.identity(beta_lasso.shape[0])))

        print "got here 2"

        beta_ols = (Xpp.transpose()).dot(inv(V)).dot(Xpp) + 0.001 * np.identity(beta_ols.shape)
        beta_ols = np.linalg.inv(beta_ols)
        beta_ols = beta_ols.dot(Xpp.transpose()).dot(np.linalg.inv(V)).dot(Y)
        beta_lasso = np.multiply(np.sign(beta_ols), threshold(np.absolute(beta_ols)-lambda1,0.0))

        print "got here 3"

        X_Xpp_sub = sftdiff = np.where(M.isnull(), X, Xpp)
        u = u + rho*(Xpp - X)
        
        i += 1

        print "-"*20
        print i
        print np.linalg.norm(Xp-X_Xp_sub)/np.linalg.norm(Xp)
        print np.linalg.norm(Xpp-X_Xpp_sub)/np.linalg.norm(Xpp)
        pprint(beta_ols)


def main(args):

    """
    X,Y,U,G = get_matrices('Height')
    X_new = DataFrame(run_softimpute(X))

    run_lmm(X_new, Y, U, G)
    """
    X,Y,U,G = get_matrices('Height')
    run_admm(X,Y,U,G)

if __name__ == '__main__':
    main(sys.argv[1:])
