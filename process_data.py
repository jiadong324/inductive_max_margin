'''
    Generate expression data for 593 breast cancer samples, where there are 62 positive samples The size of the expression matrix is going to be 16146*593.
'''
import numpy as np

data_path = "database/"

# Extract the gene entrez_id to build the expression data. There are actually 17814 different genes in the original expression data, only 16146 of them we have corresponding
# entrez_id
file = np.loadtxt(data_path+"FILE_SAMPLE_MAP.txt",dtype = str, delimiter='\t')
filename = file[1:,0]
sample_code = file[1:,1]
expression_path = data_path + "Expression-Genes/UNC__AgilentG4502A_07_3/Level_3/"
gene = np.loadtxt(expression_path+filename[0],dtype = str, delimiter = '\t') [2:,0]
geneID_symbol = np.loadtxt("geneID_symbol.txt",dtype=str,delimiter = '\t')[30:,0]
gene_id = []
gene_symbol = []
for element in geneID_symbol:
    tmp = element.split('|')
    gene_id.append(tmp[1])
    gene_symbol.append(tmp[0])

gene_id_symbol = dict(zip(gene_symbol,gene_id))

geneID = []
noID_genes_index = []
noID_genes = []
for index in range(0,len(gene)):
    if gene_id_symbol.has_key(gene[index]) == False:
        noID_genes_index.append(index)
        noID_genes.append(gene[index])
    else:
        geneID.append(gene_id_symbol[gene[index]])
# # Merge 593 samples into one expression matrix.
# ndarray_gene_expression = np.asarray(gene)[:,np.newaxis]
# 
# for sample in filename:
#     one_sample = np.loadtxt(expression_path+sample,dtype = str, delimiter = '\t')[2:,1][:,np.newaxis]
#     ndarray_gene_expression = np.hstack((ndarray_gene_expression,one_sample))
#     print ndarray_gene_expression.shape
# np.savetxt("BRC_expression_matrix.txt",ndarray_gene_expression,fmt = '%s',delimiter = '\t')

# Keep genes with entrez_id in the expression matrix
ndarray_gene_expression = np.loadtxt("BRC_expression_matrix.txt",dtype = str, delimiter = '\t')

gene_expression = np.delete(ndarray_gene_expression, noID_genes_index, 0)
ndarray_geneID = np.asarray(geneID)[:,np.newaxis]
geneID_expression = np.hstack((ndarray_geneID,gene_expression[:,1:]))
for i in range(0,geneID_expression.shape[0]):
    for j in range(0,geneID_expression.shape[1]):
        if geneID_expression[i][j] == 'null':
            geneID_expression[i][j] = '0'
        
np.savetxt("BRC_geneID_expression.txt",geneID_expression, fmt = '%s',delimiter = '\t') 

def true_label():
    index = np.arange(0,len(sample_code))
    status = []
    for code in sample_code:
        tmp = code.split('-')
        sample_status = tmp[3][0]
        if sample_status == '0':
            status.append(1)
        else:
            status.append(-1)
    sample_status_dict = dict(zip(index,status))
    return sample_status_dict


