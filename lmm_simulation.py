import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from cvxpy import *
from fancyimpute import SoftImpute
import pdb
import cPickle as pickle
import sklearn.preprocessing as skPP

# Initialize data.
n=400
m=400

np.random.seed(15)

percentMissing=0.1
X_rand = np.random.rand(n,m)
X = X_rand.copy()
for i in xrange(int(n*m*percentMissing)):
    a=np.random.randint(n); b= np.random.randint(m);
    X[a,b]=np.nan

X_NaN = X.copy()
X_NaN_loc    = np.argwhere(np.isnan(X_NaN))
X_nonNaN_loc = np.argwhere(np.invert(np.isnan(X_NaN)))

X_ref_zeroed = np.nan_to_num(X_NaN)
# for i in xrange(int(n*m*0.2)):
#     a=np.random.randint(n); b= np.random.randint(m);
#     X[a,b]=0

X_k = X_ref_zeroed.copy()
Xp_k = np.random.rand(n,m)
Xp2_k = np.random.rand(n,m)

Y = np.random.rand(n,1)
sigma = np.random.rand(n,n)
beta_k = np.random.rand(m,1)
beta_rand = np.linalg.solve(X_rand,Y)

U_k = np.zeros([n,m])
I = np.eye(n,m)

multiplier = 10
gamma1 = 0.35 * multiplier
gamma2 = 0.25 * multiplier
gamma3 = 0.08 * multiplier
rho = 10 * multiplier
alpha = 0.1 * multiplier

x_dist_old = 0
b_dist_old = 0

print "Starting"
updateVal = True
iter = 0

x_distl = []
x_xp2_distl = []
b_distl = []
real_distl = []

obj_lmm = []
obj_betareg = []
obj_xdiff = []
obj_nucnorm = []

while updateVal:
    #print "Running iteration"
    iter += 1

    if iter > 22:
        break

    X_old = Xp_k
    b_old = beta_k

    #pdb.set_trace()
    print Xp2_k.shape
    print Y.shape
    beta_ols = np.linalg.inv(Xp2_k.T.dot(np.linalg.inv(sigma)).dot(
                                Xp2_k)).dot(Xp2_k.T).dot(np.linalg.inv(sigma)).dot(Y)
    beta_k = np.multiply(np.sign(beta_ols),np.maximum(np.absolute(beta_ols)-gamma1,0.))
    beta_k = beta_k.astype('float')

    beta_k_unscaled = beta_k.copy()
    beta_k = skPP.scale(beta_k)

    # soft threshold should be iterative, with substitution for X_k
    X_k_mod = np.where(np.isnan(X_NaN),Xp_k,X_NaN)
    Xp_k_old = Xp_k.copy()
    Xpold_Xp_diff = 999
    jjj = 0
    while Xpold_Xp_diff > 0.001:
        jjj += 1
        if jjj > 10:
            break
        u,s,v = np.linalg.svd((X_k_mod*gamma2+Xp2_k*rho+U_k)/(gamma2+rho))

        #Soft Thresholding of Singular Value Decomp
        s = np.multiply(np.sign(s),np.maximum(np.absolute(s)-(gamma3/(gamma2+rho)),0.))
        x = np.zeros((n,m))
        x[:n,:n] = np.diag(s)
        s = x
        Xp_k = u.dot(s).dot(v.T)
        Xpold_Xp_diff = np.linalg.norm(Xp_k-Xp_k_old,'fro')**2/np.linalg.norm(Xp_k_old,'fro')**2
        X_k_mod = np.where(np.isnan(X_NaN),Xp_k,X_NaN)
        Xp_k_old = Xp_k.copy()

    Xp2_k = (Y.dot(beta_k.T) + rho/2.*Xp_k - U_k).dot(
                        np.linalg.inv(beta_k.dot(beta_k.T)+rho/2.*np.eye(m)))

    x_dist = np.linalg.norm((Xp_k - X_old),2)
    x_xp2_dist = np.linalg.norm((Xp2_k - Xp_k),2)
    b_dist = np.linalg.norm((beta_k_unscaled - beta_rand),2)
    real_dist = np.linalg.norm(Xp2_k-X_rand,2)

    x_distl.append(x_dist)
    x_xp2_distl.append(x_xp2_dist)
    b_distl.append(b_dist)
    real_distl.append(real_dist)

    # plots of objective function components
    obj_lmm.append(np.linalg.norm((Y-Xp2_k.dot(beta_k)),2)**2)
    obj_betareg.append(gamma1*np.linalg.norm(beta_k,1))
    obj_nucnorm.append(gamma3*np.linalg.norm(Xp_k,'nuc'))

    if np.linalg.norm((Y-Xp2_k.dot(beta_k)),2)<3e6 and real_dist<1e4:
        print "Iter: {}".format(iter)
        print "Distance between ADMM x: {}".format(x_dist)
        print "Distance between ADMM b: {}".format(b_dist)
        print "Distance between constraint: {}".format(np.linalg.norm((Y-Xp2_k.dot(beta_k)),2))
        #print "Distance between ADMM X, fancyimpute X: {}".format(np.linalg.norm(Xp2_k-X_ref,2))

        diff = 0
        for index in xrange(len(X_nonNaN_loc)):
            xy = X_nonNaN_loc[index]
            diff += (Xp2_k[xy[0],xy[1]] - X_NaN[xy[0],xy[1]])**2
        diff = np.sqrt(diff)
        print "Distance between ADMM X, nonNaN locations X: {}".format(diff)
        obj_xdiff.append(gamma2*diff)
        #print Xp2_k
        #print beta_k
        print "----"
        #pdb.set_trace()
    if x_dist<=1.0e-15 and b_dist<=1.0e-15:
    #    print "Distance between ADMM x: {}".format(x_dist)
        print "Distance between ADMM b: {}".format(b_dist)
        updateVal = False

    if x_dist_old == x_dist and b_dist_old == b_dist:
        updateVal = False
    else:
        x_dist_old = x_dist
        b_dist_old = b_dist
# print X_k
# print beta_k
# print beta_k_unscaled

plt.figure()
plt.plot(real_distl)
plt.plot(b_distl)
#plt.title("(" + str(int(percentMissing*100)) + "% missing input) " + "per-iteration L2-norm of difference in beta")
#plt.ylabel("Error")
#plt.xlabel("Iterations")
#plt.savefig(str(int(percentMissing*100))+'missing_beta.png', bbox_inches='tight')

plt.figure()
plt.plot(real_distl)
#plt.title("(" + str(int(percentMissing*100)) + "% missing input) " + "per-iteration L2-norm of difference between X'' and simulated X")
#plt.ylabel("Error")
#plt.xlabel("Iterations")
#plt.savefig(str(int(percentMissing*100))+'missing_x.png', bbox_inches='tight')
plt.show()
