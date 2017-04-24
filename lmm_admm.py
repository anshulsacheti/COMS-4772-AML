import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from cvxpy import *
from fancyimpute import SoftImpute
import pdb

# Initialize data.
n=10
m=10

X_ref = np.random.rand(n,n)
X = X_ref.copy()
for i in xrange(n*2):
    a=np.random.randint(n); b= np.random.randint(n);
    X[a,b]=0

X_k = X.copy()
Xp_k = np.random.rand(n,n)
Xp2_k = np.random.rand(n,n)

Y = np.random.rand(n,1)
sigma = np.identity(n)
beta_k = np.random.rand(n,1)

U_k = np.zeros([n,n])
I = np.identity(n)
gamma1 = 0.15
gamma2 = 0.25
gamma3 = 0.1
rho = 0.1
alpha = 0.1


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

while updateVal:
    #print "Running iteration"
    X_old = Xp_k
    b_old = beta_k

    #pdb.set_trace()
    beta_ols = np.linalg.inv(Xp2_k.T.dot(np.linalg.inv(sigma)).dot(
                                Xp2_k)).dot(Xp2_k.T).dot(np.linalg.inv(sigma)).dot(Y)
    beta_k = np.multiply(np.sign(beta_ols),np.maximum(np.absolute(beta_ols)-gamma1,0.))

    u,s,v = np.linalg.svd((X_k*gamma2+Xp2_k*rho+U_k)/(gamma2+rho))

    #Soft Thresholding of Singular Value Decomp
    s = np.multiply(np.sign(s),np.maximum(np.absolute(s)-(gamma3/(gamma2+rho)),0.))
    x = np.zeros((m,n))
    x[:m,:m] = np.diag(s)
    s = x

    Xp_k = u.dot(s).dot(v.T)

    Xp2_k = (Y.dot(beta_k.T) + rho/2.*Xp_k - U_k).dot(np.linalg.inv(
                    beta_k.dot(beta_k.T)+rho/2.*np.eye(n)))

    #U_k = U_k + alpha*(Xp_k-Xp2_k)
    # #pdb.set_trace()
    #Final update
    #U_k = U_k + alpha*(Y-Xp2_k.dot(beta_k))

    # print("Sum of Xp2_k: %d" % (np.sum(Xp2_k)))
    # print("Sum of beta_ols: %d" % (np.sum(np.abs(beta_ols))))
    # print("Sum of beta_k: %d" % (np.sum(np.abs(beta_k))))
    x_dist = np.linalg.norm((Xp_k - X_old),2)
    b_dist = np.linalg.norm((beta_k - b_old),2)
    real_dist = np.linalg.norm(Xp2_k-X_ref,2)
    if np.linalg.norm((Y-Xp2_k.dot(beta_k)),2)<4. and real_dist<6:
        print "Distance between ADMM x: {}".format(x_dist)
        print "Distance between ADMM b: {}".format(b_dist)
        print "Distance between constraint: {}".format(np.linalg.norm((Y-Xp2_k.dot(beta_k)),2))
        print "Distance between real X: {}".format(np.linalg.norm(Xp2_k-X_ref,2))
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
print X_k
print beta_k
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
