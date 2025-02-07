import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from cvxpy import *
from fancyimpute import SoftImpute
import pdb
import cPickle as pickle
import sklearn.preprocessing as skPP

def normalizeWithNan(ary):
    mean = np.nanmean(ary, axis=0)
    stdd = np.nanstd(ary, axis=0)
    ary = ary - mean
    ary = ary * 1/stdd
    return ary

# Initialize data.
n=435
m=2707
np.random.seed(15)

X_ref = pickle.load(open("X.p", "rb")).as_matrix().astype("float64")
X_NaN = X_ref.copy()
X_nonNaN_loc = np.argwhere(np.invert(np.isnan(X_NaN)))

X_ref_zeroed = np.nan_to_num(X_ref)
X_ref = np.load("X_softImputed.npy").astype("float64")
# for i in xrange(int(n*m*0.2)):
#     a=np.random.randint(n); b= np.random.randint(m);
#     X[a,b]=0

X_k = X_ref_zeroed.copy()
Xp_k = np.random.rand(n,m)
Xp2_k = np.random.rand(n,m)

Y = pickle.load(open("Y.p", "rb")).as_matrix(columns=["Height"]).astype("float64")
Y = normalizeWithNan(Y)
sigma = pickle.load(open("V.p", "rb")).as_matrix().astype("float64")
sigma = normalizeWithNan(sigma)
beta_k = np.random.rand(m,1)

U_k = np.zeros([n,m])
I = np.eye(n,m)

multiplier = 10
gamma1 = 1.5 * multiplier
gamma2 = 0.25 * multiplier
gamma3 = 0.08 * multiplier
rho = 1.0 * multiplier
alpha = 0.1 * multiplier


#CVXPY Solution
#   min X,B ||y-Xp*B||^2 + lambda_1 * ||B||_1 + lambda_2/2 * ||Xp-X||^2 + lambda_3 * ||Xp||_*
#   s.t. Xp=X
# Initialize problem.
# b = Variable(n,1)
# vXp = Variable(n,n)
# fX = norm(vXp, "nuc")
# fB = norm(b, 1)
# fS = sum_squares(Y-X_k*b)
# fXp = norm(vXp-X, 2)
#
# lagr = fS + gamma1*fB + 16.*gamma2/2.*fXp + gamma3*fX
# Problem(Minimize(lagr)).solve()
# vXp.value[0]
# X_ref[0]
# fXp.value
#ADMM Converged Solution

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

    if iter > 10:
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

    #U_k = alpha*(Xp_k-Xp2_k)
    # #pdb.set_trace()
    #Final update
    #U_k = U_k + alpha*(Y-Xp2_k.dot(beta_k))

    # print("Sum of Xp2_k: %d" % (np.sum(Xp2_k)))
    # print("Sum of beta_ols: %d" % (np.sum(np.abs(beta_ols))))
    # print("Sum of beta_k: %d" % (np.sum(np.abs(beta_k))))
    x_dist = np.linalg.norm((Xp_k - X_old),2)
    x_xp2_dist = np.linalg.norm((Xp2_k - Xp_k),2)
    b_dist = np.linalg.norm((beta_k - b_old),2)
    real_dist = np.linalg.norm(Xp2_k-X_ref,2)

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
        print "Distance between ADMM X, fancyimpute X: {}".format(np.linalg.norm(Xp2_k-X_ref,2))

        diff = 0
        for index in xrange(len(X_nonNaN_loc)):
            xy = X_nonNaN_loc[index]
            diff += (Xp2_k[xy[0],xy[1]] - X_NaN[xy[0],xy[1]])**2
        diff = np.sqrt(diff)
        print "Distance between ADMM X, nonNaN locations X: {}".format(diff)
        obj_xdiff.append(gamma2*diff)
        #tmp = np.ma.array(X_NaN, mask=np.isnan(X_NaN)) # Use a mask to mark the NaNs
        #tmp1 = np.linalg.norm(tmp[~tmp.mask],1)
        #tmp2 = np.linalg.norm(tmp[~tmp.mask],2)
        #print "l1 norm percentage X\'\'/X_NaN : {}".format(np.linalg.norm(Xp2_k,1)/tmp1)
        #print "l2 norm percentage X\'\'/X_NaN : {}".format(np.linalg.norm(Xp2_k,2)/tmp2)

        print "Distance between real B: {}"
        #print Xp2_k
        print beta_k
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

plt.plot(x_distl)
plt.savefig('snp_perIterXdiff.png', bbox_inches='tight')
# plt.title("per-iteration L2-norm of difference in imputed X'")
# plt.ylabel("L2-Diff")
# plt.xlabel("Iterations")

#plt.figure()
#plt.plot(x_xp2_distl)
# plt.title("per-iteration L2-norm of difference between X' and X''")
# plt.ylabel("L2-Diff")
# plt.xlabel("Iterations")

plt.figure()
plt.plot(b_distl)
plt.savefig('snp_perIterBdiff.png', bbox_inches='tight')
# plt.title("per-iteration L2-norm of difference in beta")
# plt.ylabel("L2-Diff")
# plt.xlabel("Iterations")

plt.figure()
plt.plot(real_distl)
plt.savefig('snp_Xdiff.png', bbox_inches='tight')
# plt.title("per-iteration L2-norm of difference between X'' and original X")
# plt.ylabel("L2-Diff")
# plt.xlabel("Iterations")

# plots of objective function components
plt.figure()
plt.plot(obj_lmm)
plt.savefig('snp_constraintError.png', bbox_inches='tight')
#plt.plot(obj_betareg)
#plt.plot(obj_xdiff)
#plt.plot(obj_nucnorm)
#plt.legend(['Y-XB', 'B_1', "X'-X", "X'_*"])
#plt.title("objective function additive components")
plt.show()

print X_k
print beta_k
print beta_k_unscaled
#--------------------------------------------
#Reference to test against
# X = X_ref.copy()
# for i in xrange(n*2):
#     a=np.random.randint(n); b= np.random.randint(n);
#     X[a,b]=np.nan
# X_si = SoftImpute().complete(X)
#
# x_dist = np.linalg.norm((X_si - X_k),2)
#b_dist = np.linalg.norm((beta_k - S.value),2)
print "Distance between X: {}".format(x_dist)
#print "Distance between b: {}".format(b_dist)
#print "S:   --------------------\n{}".format(S.value)
#print "S_k: --------------------\n{}".format(S_k)
#print "L:   --------------------\n{}".format(L.value)
#print "L_k: --------------------\n{}".format(L_k)
