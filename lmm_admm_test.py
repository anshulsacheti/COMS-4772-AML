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
Xp_k = X.copy()
Xp2_k = np.random.rand(n,n)

h_k = np.random.rand(n,n)
g_k = np.random.rand(n,1)
Y = np.random.rand(n,1)
sigma = np.identity(n)
beta_k = np.random.rand(n,1)

U_k = np.zeros([n,n])
I = np.identity(n)
gamma1 = 0.75
gamma2 = 0.15
gamma3 = 0.1
rho = 0.1

multiplier = 1.0
alpha = 0.75 * multiplier
z = 0.1 * multiplier
gamma = 0.01 * multiplier
nu = 0.01 * multiplier
eta = 0.08 * multiplier

lambda_k = np.zeros([n,1])
mu_k = np.zeros([n,1])
rho_k = np.zeros([n,n])
theta_k = np.zeros([n,n])

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

    h_k = (gamma*Xp_k + rho_k + z*Y.dot(g_k.T) + lambda_k.dot(g_k.T)).dot(
                                    np.linalg.inv(gamma*I + z*g_k.dot(g_k.T)))

    g_k = np.linalg.inv(alpha*I + z*h_k.T.dot(h_k)).dot(
                    alpha*beta_k + mu_k + z*h_k.T.dot(Y) + h_k.T.dot(lambda_k))

    beta_ols = g_k - 1./alpha*mu_k
    beta_k = np.multiply(np.sign(beta_ols),np.maximum(np.absolute(beta_ols)-nu/alpha,0.))

    #Soft Thresholding of Singular Value Decomp
    # u,s,v = np.linalg.svd(h_k+1./alpha*rho_k)
    # #print s
    # s = np.multiply(np.sign(s),np.maximum(np.absolute(s)-(1./alpha),0.))
    # #print s
    # x = np.zeros((m,n))
    # x[:m,:m] = np.diag(s)
    # s = x
    # Xp_k = u.dot(s).dot(v.T)

    u,s,v = np.linalg.svd((gamma*h_k+eta*X_k+rho_k)/(gamma+eta))
    #print s
    s = np.multiply(np.sign(s),np.maximum(np.absolute(s)-(gamma+eta),0.))
    #print s
    x = np.zeros((m,n))
    x[:m,:m] = np.diag(s)
    s = x
    Xp_k = u.dot(s).dot(v.T)

    #Xp_k = X_k
    #print Xp_k
    lambda_k = lambda_k + z*(Y-h_k.dot(g_k))
    rho_k = rho_k + gamma*(Xp_k-h_k)
    mu_k = mu_k + alpha*(beta_k-g_k)
    theta_k = theta_k + eta*(Xp_k - X_k)

    #U_k = U_k + alpha*(Xp_k-Xp2_k)
    # #pdb.set_trace()
    #Final update
    #U_k = U_k + alpha*(Y-Xp2_k.dot(beta_k))

    # print("Sum of h_k: %d" % (np.sum(h_k)))
    # print("Sum of beta_ols: %d" % (np.sum(np.abs(beta_ols))))
    # print("Sum of beta_k: %d" % (np.sum(np.abs(beta_k))))
    x_dist = np.linalg.norm((Xp_k - X_old),2)
    b_dist = np.linalg.norm((beta_k - b_old),2)
    # print "Distance between ADMM x: {}".format(x_dist)
    # print "Distance between ADMM b: {}".format(b_dist)
    # print "Distance between constraint: {}".format(np.linalg.norm((Y-Xp_k.dot(beta_k)),2))
    # print "Distance between real X: {}".format(np.linalg.norm(Xp_k-X_ref,2))
    # #print beta_k
    # print "----"
    real_dist = np.linalg.norm(Xp_k-X_ref,2)
    if np.linalg.norm((Y-Xp_k.dot(beta_k)),2)<4. and real_dist<2.6:
        print "Distance between ADMM x: {}".format(x_dist)
        print "Distance between ADMM b: {}".format(b_dist)
        print "Distance between constraint: {}".format(np.linalg.norm((Y-Xp_k.dot(beta_k)),2))
        print "Distance between real X: {}".format(np.linalg.norm(Xp_k-X_ref,2))
        print X_ref[0]
        print Xp_k[0]
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
#print X_k
#print beta_k
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
