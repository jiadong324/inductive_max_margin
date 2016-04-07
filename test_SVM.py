import numpy as np

from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pylab as plt
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split

import process_data

sample_status_dict = process_data.true_label()
true_label = sample_status_dict.values()
true_label = np.asarray(true_label)

# Breast cancer gene expression matrix
expression_matrix = np.loadtxt("BRC_geneID_expression.txt", dtype = str, delimiter = '\t')
expression = expression_matrix.astype(np.float)

breast_cancer_geneID = ['8438','841','580','5290','3161','4835','2099','9821','5002','7251','472','3845','675','7517','207','5888',
                        '79728','999','7157','5245','8493','83990','112000']

gene_index = []

for val in breast_cancer_geneID:
    index = np.where(expression_matrix[:,0]==val)[0]
    if index.shape[0]!=0:
        gene_index.append(index[0])
    
# expression_matrix = stats.zscore(expression_matrix)
expression = expression[gene_index]
expression = np.transpose(preprocessing.normalize(np.transpose(expression[:,1:])))
X = np.transpose(expression)
X = np.hstack((true_label[:,np.newaxis],X))

shuffled_X = shuffle(X)


skf = StratifiedKFold(X[:,0],n_folds=3)

for i, (train_idx, test_idx) in enumerate(skf):
#     print("TRAIN:", train_idx.shape, "TEST:", test_idx.shape)
    X_train,X_test = shuffled_X[:,1:][train_idx],shuffled_X[:,1:][test_idx]
    y_train,y_test = shuffled_X[:,0][train_idx],shuffled_X[:,0][test_idx]
    svm = SVC(kernel='linear')
    y_pred = svm.fit(X_train,y_train).predict(X_test)
    fpr,tpr,thresholds = metrics.roc_curve(y_test,y_pred)
    auc = metrics.auc(fpr,tpr)
    weights = svm.coef_
    plt.plot(fpr,tpr,lw=1,label='ROC fold %d (area = %0.2f)' % (i, auc))
    
plt.plot([0,1],[0,1],'b-',label='Random')
plt.legend(loc=0)
plt.show()
