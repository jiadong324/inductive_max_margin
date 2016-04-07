import numpy as np
from sklearn import metrics
import matplotlib.pylab as plt
from sklearn import preprocessing
from sklearn.linear_model import ElasticNet
from sklearn.cross_validation import StratifiedKFold


import process_data
import gradient_descent

def inductive_max_margin_predict(E,alpha1,alpha2,lambda1,lambda2,max_iter):
    '''
    The inductive max margin algorithm. 
    alpha1 and alpha2 are learning rate.
    lambda1 and lambda2 are penalty coefficient.
    Return the predicted label for samples in expression data.
    '''
    
    W,beta = gradient_descent.gradient_solver(lambda1, lambda2, alpha1, alpha2, max_iter)
    y_predict = np.dot(np.transpose(W),E)  
    return y_predict

def draw_ROC(y_predict,y_true):
    fpr,tpr,thresholds = metrics.roc_curve(y_true,y_predict)
    auc = metrics.auc(fpr,tpr)
    plt.plot(fpr,tpr,lw=1,label = 'ROC (area = %0.2f)' % auc)
    plt.plot([0,1],[0,1],'--',color='b',label='random')
    plt.show()
# def confusion_matrix(Y,y_predict):
#     positive_sample_index = []
#     negative_sample_index = []
#     for i in range(0,len(Y)):
#         if Y[i] == 0:
#             negative_sample_index.append(i)
#         else:
#             positive_sample_index.append(i)
#     true_positive = []
#     true_negative = []
#     false_positive = []
#     false_negative = []
#     for idx in positive_sample_index:
#         if y_predict[idx] == 1:
#             true_positive.append(idx)
#         else:
#             false_positive.append(idx)
#     for idx in negative_sample_index:
#         if y_predict[idx] == 0:
#             true_negative.append(idx)
#         else:
#             false_negative.append(idx)
#     matrix = np.zeros([2,2])
#     matrix[0][0] = len(true_positive)
#     matrix[0][1] = len(false_positive)
#     matrix[1][1] = len(true_negative)
#     matrix[1][0] = len(false_negative)
#     
#     return matrix
    
    
if __name__ == 'main':
    sample_status_dict = process_data.true_label()
    true_label = sample_status_dict.values()
    true_label = np.asarray(true_label)[:,np.newaxis]
    
    # Breast cancer gene expression matrix
    expression_matrix = np.loadtxt("BRC_geneID_expression.txt", dtype = str, delimiter = '\t')[:,1:]
    expression_matrix = expression_matrix.astype(np.float)
    # expression_matrix = stats.zscore(expression_matrix)
    expression_matrix = np.transpose(preprocessing.normalize(np.transpose(expression_matrix)))
    X = expression_matrix
    skf = StratifiedKFold(true_label,n_folds=3)
    
    for train_idx, test_idx in skf:
        X_train,X_test = X[train_idx],X[test_idx]
        y_train,y_test = true_label[train_idx],true_label[test_idx]
        y_pred = ElasticNet.fit(X_train,y_train).predict(X_test)
        
        