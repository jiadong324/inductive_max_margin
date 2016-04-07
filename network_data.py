'''
    Generate a matrix representation of the gene-gene interaction network
'''
import numpy as np
from scipy.sparse import linalg
import cPickle as pickle

# f = np.loadtxt("HumanNet.v1.join.txt",dtype = str, delimiter = '\t')
#    
# column_0 = np.asarray(f[:,0])[:,np.newaxis]
# column_1 = np.asarray(f[:,1])[:,np.newaxis]
# network_gene = np.hstack((column_0,column_1))
# network_geneIDs = []
# for val in f[:,0]:
#     if val not in network_geneIDs:
#         network_geneIDs.append(val)
#        
# for val in f[:,1]:
#     if val not in network_geneIDs:
#         network_geneIDs.append(val)
#   
# print "-----Network geneIDs generated-----"
#   
# expression_data = np.loadtxt("BRC_geneID_expression.txt",dtype = str, delimiter = '\t')
# expression_geneIDs = expression_data[:,0]
#   
# tuple_pair_genes = []
# for pair in network_gene:
#     tup = (pair[0],pair[1])
#     tuple_pair_genes.append(tup)
# weight = f[:,23]   
# genes_weight_dict = dict(zip(tuple_pair_genes,weight))
#     
# index = np.arange(0,len(expression_geneIDs))
#    
# geneID_map = dict(zip(index,expression_geneIDs))    
#   
# print "-----Start to build matrix------"  
# network_matrix = np.zeros([len(expression_geneIDs),len(expression_geneIDs)])
#      
# for i in range(0,len(expression_geneIDs)):
#     for j in range(0,len(expression_geneIDs)):
#         gene_i = geneID_map[i]
#         gene_j = geneID_map[j]
#         tup = (gene_i,gene_j)
#         if genes_weight_dict.has_key(tup) == True:
#             val = genes_weight_dict[tup]
#             network_matrix[i][j] = val
#              
# sparse_network_matrix = sparse.csc_matrix(network_matrix)
#  
# with open('sparse_gene_expression.dat','wb') as outfile:
#     pickle.dump(sparse_network_matrix,outfile,pickle.HIGHEST_PROTOCOL)
#     outfile.close()   
with open('sparse_gene_expression.dat','rb') as infile:
    x = pickle.load(infile)
U,s,V = linalg.svds(x,k=1000)
print U.shape,s,V.shape
