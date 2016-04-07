import numpy as np
import process_data
import cPickle as pickle
from sklearn import preprocessing


def w_derivative(X,F,Y,W0,beta0,lemda1,lemda2):
    '''
    derivative of the objective function
    '''
    vector1 = []
    for i in range(0,X.shape[0]):
        vector1.append(-np.dot(F[i],beta0)/float(F.shape[0]))
    vector1 = np.asarray(vector1)
    sample = np.dot(np.transpose(X),W0)
    W_deriviation = 2*(np.dot(X,sample)-np.dot(X,Y))/float(X.shape[1])+lemda1*np.sign(W0)-lemda2*vector1
    
    return W_deriviation

def w(X,F,Y,theta,W0,beta0,lemda1,lemda2):
    '''
    w is the objective function
    '''
    function = np.sum([(Y[i]-np.dot(np.transpose(W0),X[:,i]))**2 for i in range(0,X.shape[1])])/float(X.shape[1])+lemda1*np.linalg.norm(W0, ord=1)
    penalty = 0
    for i in range(0,F.shape[0]):
        tmp = 1 - (W0[i]-theta)*np.dot(F[i],beta0)
        val = max(0,tmp)     
        penalty += val
    penalty = penalty/float(F.shape[0])
    value = function+lemda2*penalty
    
    return value

def backtrack_search_W(X,F,Y,theta,W0,beta0,lemda1,lemda2,alpha,m):
    '''
    alpha: learning rate
    m: A parameter for updating alpha
    '''
#     print '-----starting search W'
    delta_w = w_derivative(X, F, Y, W0, beta0, lemda1, lemda2)
    temp = 0
    for i in range(0,len(delta_w)):
        temp+=(delta_w[i])**2
    val = w(X,F,Y,theta,W0-alpha*delta_w,beta0,lemda1,lemda2) - w(X,F,Y,theta,W0,beta0,lemda1,lemda2) + (0.3*alpha)*temp
    print val
    while w(X,F,Y,theta,W0-alpha*delta_w,beta0,lemda1,lemda2) > w(X,F,Y,theta,W0,beta0,lemda1,lemda2) - (0.3*alpha)*temp:
        alpha = m*alpha
    print '----W backtracked alpha----',alpha
    return alpha

def beta_derivative(X,F,Y,theta,W0,beta0,lemda):
    update_vector = np.zeros([F.shape[1],1])    
    for i in range(0,F.shape[0]):
        g = 1-(W0[i]-theta)*np.dot(F[i],beta0)
        sign = (np.sign(g)+1)/2
        temp = -(sign*(W0[i]-theta)*np.transpose(F[i])[:,np.newaxis])/float(F.shape[0])
        update_vector = update_vector+temp
    beta_vector = lemda*update_vector
    return beta_vector

def backtrack_search_beta(X,F,Y,theta,W0,beta0,lemda1,lemda2,alpha,m):
    '''
    alpha: learning rate
    m: A parameter for updating alpha
    '''
#     print '-----starting search beta'
    delta_beta = beta_derivative(X, F, Y, theta, W0, beta0, lemda2)
    temp = 0
    for i in range(0,len(delta_beta)):
        temp+=(delta_beta[i])**2
#     val = w(X,F,Y,theta,W0,beta0-alpha*delta_beta,lemda1,lemda2) - w(X,F,Y,theta,W0,beta0,lemda1,lemda2) + (alpha/2)*temp
    
    while w(X,F,Y,theta,W0,beta0-alpha*delta_beta,lemda1,lemda2) > w(X,F,Y,theta,W0,beta0,lemda1,lemda2) + (0.3*alpha)*temp:
        alpha = m*alpha
    print '----beta backtracked alpha----',alpha
    return alpha

def W_check_alpha(X,F,Y,theta,W0,beta0,lemda1,lemda2,alpha,m):
    delta_w = w_derivative(X, F, Y, W0, beta0, lemda1, lemda2)
    temp = 0
    for i in range(0,len(delta_w)):
        temp+=(delta_w[i])**2
    val = w(X,F,Y,theta,W0-alpha*delta_w,beta0,lemda1,lemda2) - w(X,F,Y,theta,W0,beta0,lemda1,lemda2) + (alpha/2)*temp
    
    return val

def beta_check_alpha(X,F,Y,theta,W0,beta0,lemda1,lemda2,alpha,m):
    delta_beta = beta_derivative(X, F, Y, theta, W0, beta0, lemda2)
    temp = 0
    for i in range(0,len(delta_beta)):
        temp+=(delta_beta[i])**2
    val = w(X,F,Y,theta,W0,beta0-alpha*delta_beta,lemda1,lemda2) - w(X,F,Y,theta,W0,beta0,lemda1,lemda2) + (alpha/2)*temp
    
    return val

def choose_theta(W,distribution):
    sorted_W = np.sort(W,axis=0)
    index = distribution*sorted_W.shape[0]
    theta = sorted_W[int(index)]
    
    return theta
        