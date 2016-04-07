import numpy as np

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import metrics
import matplotlib.pylab as plt
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils import shuffle

import process_data

sample_status_dict = process_data.true_label()
true_label = sample_status_dict.values()
true_label = np.asarray(true_label)

# Breast cancer gene expression matrix
expression_matrix = np.loadtxt("BRC_geneID_expression.txt", dtype = str, delimiter = '\t')

breast_cancer_geneID = ['8438','841','580','5290','3161','4835','2099','9821','5002','7251','472','3845','675','7517','207','5888',
                        '79728','999','7157','5245','8493','83990','112000']

BRC_gene_index = []

for val in breast_cancer_geneID:
    index = np.where(expression_matrix[:,0]==val)[0]
    if index.shape[0]!=0:
        BRC_gene_index.append(index[0])

# expression_matrix = np.delete(expression_matrix,BRC_gene_index,0)
# expression_matrix = expression_matrix[BRC_gene_index]
expression = expression_matrix[:,1:].astype(np.float)
# expression = preprocessing.StandardScaler().fit_transform(np.transpose(expression))
expression = preprocessing.normalize(np.transpose(expression),norm='l2')
X = expression
X = np.hstack((true_label[:,np.newaxis],X))
shuffled_X = shuffle(X)
skf = StratifiedKFold(shuffled_X[:,0],n_folds=3)


for i, (train_idx, test_idx) in enumerate(skf):
#     print("TRAIN:", train_idx.shape, "TEST:", test_idx.shape)
    X_train,X_test = shuffled_X[:,1:][train_idx],shuffled_X[:,1:][test_idx]
    y_train,y_test = shuffled_X[:,0][train_idx],shuffled_X[:,0][test_idx]
#     clf = ElasticNet(alpha=1.1,l1_ratio=0.8,max_iter=600)
#     clf = Lasso()
#     clf = LogisticRegression(penalty='l2')
    clf = LinearSVC()
    y_pred = clf.fit(X_train,y_train).predict(X_test)
    fpr,tpr,thresholds = metrics.roc_curve(y_test,y_pred)
    auc = metrics.auc(fpr,tpr)
    weights = clf.coef_
    accuracy = clf.score(X_test,y_test)
    plt.plot(fpr,tpr,lw=1,label='ROC fold %d (area = %0.2f)' % (i, auc))
    
plt.plot([0,1],[0,1],'b-',label='Random')
plt.legend(loc=0)
plt.show()
