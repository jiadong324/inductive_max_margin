import numpy as np
import process_data
import cPickle as pickle
from sklearn import preprocessing
import line_search

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
W0 = np.random.random([X.shape[0],1])
beta0 = np.random.random([F.shape[1],1])
iter = 0
delta_beta0 = np.ones([X.shape[0],1])/10
delta_w = np.ones([X.shape[0],1])/1000
while iter < 10:
    
    F_beta = line_search.w(X, F, Y, 0.1, W0, beta0, 1, 1)    
    F_delta = line_search.w(X, F, Y, 0.1 ,W0, beta0+delta_beta0, 1, 1)
    F_delta_beta = line_search.beta_derivative(X, F, Y, 0.01,W0, delta_beta0,1)
    norm2 = np.linalg.norm(delta_w, ord=1, axis=0)
    val = F_delta - F_beta - np.dot(np.transpose(F_delta_beta),delta_w)
    res = val[0][0]/norm2[0]
    delta_w = 0.1*delta_w
    iter+=1
    print res

