import numpy as np
import process_data
import cPickle as pickle
import matplotlib.pylab as plt
from sklearn import preprocessing
import math
import line_search

def h_function(W,x):
    value = float(1)/(1+math.exp(np.dot(np.transpose(W),x)))
    
    return value
    
def error_function(W,beta,X,F,Y,lemda1,lemda2,theta):
    function = np.sum([(Y[i]-np.dot(np.transpose(W),X[:,i]))**2 for i in range(0,expression_matrix.shape[1])])/float(expression_matrix.shape[1])+lemda1*np.linalg.norm(W, ord=1)
#     logistic_regression_loss = -np.sum([(Y[i]*math.log(h_function(W, X[:,i]))+(1-Y[i])*math.log(1-h_function(W, X[:,i]))) for i in range(0,expression_matrix.shape[1])])/expression_matrix.shape[1]                                                                                 
#     regularization = lemda1*np.linalg.norm(W, ord=1)
#     logistic_regression_loss = 0
#     for i in range(0,expression_matrix.shape[1]):
#         logistic_regression_loss+=Y[i]*math.log(h_function(W,X[:,i]))+(1-Y[i])*math.log(1-h_function(W, X[:,i]))
#     logistic_error = -logistic_regression_loss/expression_matrix.shape[1]
#     function = logistic_error+regularization 
    penalty = 0
    for i in range(0,F.shape[0]):
        tmp = 1 - (W[i]-theta)*np.dot(F[i],beta)
        val = max(0,tmp)     
        penalty += val   
    penalty = penalty/float(F.shape[0])
    error = function+lemda2*penalty[0]
#     print 'mean squared error',np.sum([(Y[i]-np.dot(np.transpose(W),X[:,i]))**2 for i in range(0,expression_matrix.shape[1])])/float(expression_matrix.shape[1])
#     print 'l1-norm',np.linalg.norm(W, ord=1)
#     print 'hinge loss',penalty[0]
    
    return error
        
def gradient_solver(lemda1,lemda2,a1,a2,max_iter,plot):
    converged = False
    '''
        W: initial coefficient vector for expression_matrix.
        theta: a threshold to help determine the pseudo-label.
        beta: initial coefficient vector for gene_feature_matrix.
        lemda1: penalize a l1-norm.
        lemda2: penalize the constrain from the gene_feature_matrix.
        a1,a2: learning rate for gradient descent.
    '''
    X = expression_matrix
    F = gene_feature_matrix
    Y = true_label 
    iter = 1
    # Initial W and beta
    W0 = np.random.random([X.shape[0],1])
#     W0 = preprocessing.normalize(W0,norm='l1',axis=0)
    theta0 = line_search.choose_theta(W0, 0.4)[0]
    print '----chosen theta----',theta0
    
    beta0 = np.random.random([F.shape[1],1])
#     beta0 = preprocessing.normalize(beta0, norm='l1', axis=0)
    
    error0 = error_function(W0,beta0,X, F, Y, lemda1, lemda2, theta0)
    print '----Initial error-----',error0
   
    list_errors = []
    while iter <= max_iter:
        list_errors.append(error0)
        print '\n'+'----current iteration times----',iter
        
        #Updating W
#         while np.linalg.norm(line_search.w_derivative(X, F, Y, W0, beta0, lemda1, lemda2),ord=2) > 1e-2:
        a1 = line_search.backtrack_search_W(X, F, Y, theta0, W0, beta0, lemda1, lemda2, a1, 0.15)
        W = W0-a1*line_search.w_derivative(X, F, Y, W0, beta0, lemda1, lemda2)
        W0 = W
        theta = line_search.choose_theta(W0, 0.4)[0]
        theta0 = theta
        #Updating beta
#         update_vector = np.zeros([F.shape[1],1])    
#         for i in range(0,F.shape[0]):
#             g = 1-(W0[i]-theta)*np.dot(F[i],beta0)
#             sign = (np.sign(g)+1)/2
#             temp = (sign*(W0[i]-theta)*np.transpose(F[i])[:,np.newaxis])/float(F.shape[0])
#             update_vector = update_vector+temp
#         beta_vector = lemda2*update_vector
#         while np.linalg.norm(line_search.beta_derivative(X, F, Y, theta, W0, beta0, lemda2),ord=2) > 1e-2:
        a2 = line_search.backtrack_search_beta(X, F, Y, theta0, W0, beta0, lemda1, lemda2, a2, 0.5)
        beta = beta0 - a2*line_search.beta_derivative(X, F, Y, theta0, W, beta0, lemda2)
        beta0 = beta 
        #Calculate a new error
        error = error_function(W0, beta0, X, F, Y, lemda1, lemda2, theta0)
        
        print '----current error-----',error
        
        error0 = error
        iter+=1
        
    
    if plot == True:
        plt.title('Converge Status'+'\n'+'theta(%): '+str(40)+' lambda1: '+str(lemda1)+' lambda2: '+str(lemda2)+'\n'
                  +'initial alpha1: '+str(a1)+' initial alpha2: '+str(a2))
        plt.plot(list_errors,'r',label='Total loss')
        plt.legend(loc=0)
        plt.show() 
    return W0,beta0



if __name__ == '__main__':
    
    sample_status_dict = process_data.true_label()
    true_label = sample_status_dict.values()
    true_label = np.asarray(true_label)[:,np.newaxis]
    
    # Breast cancer gene expression matrix
    expression_matrix = np.loadtxt("BRC_geneID_expression.txt", dtype = str, delimiter = '\t')[:,1:]
    expression_matrix = expression_matrix.astype(np.float)
    # expression_matrix = stats.zscore(expression_matrix)
    expression_matrix = np.transpose(preprocessing.normalize(np.transpose(expression_matrix)))
    
    
    # Gene feature matrix extracted from gene-gene interaction network, each gene corresponding to the gene in the expression matrix.
    with open('sparse_gene_expression.dat','rb') as infile:
        x = pickle.load(infile)
    # gene_feature_matrx is a sparse matrix.
    gene_feature_matrix = x.toarray()
     
    X = expression_matrix
    F = gene_feature_matrix
    Y = true_label 
    
    a1 = 1
#     a1 = 0.3
#     a2 = 1/float(gene_feature_matrix.shape[0])
    a2 = 5
    
    print '----initial learning rate----',a1,a2
    
    W0,beta0 = gradient_solver(0.001, 3, a1, a2, 600, plot=True)

    
    
    
       
        