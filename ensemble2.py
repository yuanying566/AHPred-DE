import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn import metrics
import matplotlib.pylab as plt
from sklearn.ensemble import RandomForestClassifier
#from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.svm import SVC
from itertools import product
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import os
import random
from sklearn.model_selection import train_test_split
os.system("ls ../input")
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C# REF就是高斯核函数#from xgboost import XGBClassifier
from sklearn.gaussian_process import GaussianProcessClassifier


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
#def PolynomialLogisticRegression(degree):
 #   return Pipeline([
 #       ('poly',PolynomialFeatures(degree=degree)),
 #       ('std_scaler',StandardScaler()),
 #       ('log_reg',LogisticRegression())
 #   ])
def PolynomialLogisticRegression(degree,C,penalty='l2'):
    return Pipeline([
        ('poly',PolynomialFeatures(degree=degree)),
        ('std_scaler',StandardScaler()),
        ('log_reg',LogisticRegression(C=C,penalty=penalty))
    ])



X_train=pd.read_csv('/Users/yoland/PycharmProjects/untitled4/embed_HT_tr._1900.csv',header=None)
y_train=pd.read_table('/Users/yoland/PycharmProjects/untitled4/data/PNHT.tr.label.txt',header=None)
X_test=pd.read_csv('/Users/yoland/PycharmProjects/untitled4/embed_HT_te_1900.csv',header=None)


y_train=(y_train+1)/2

print(X_train.shape)
print(y_train.shape)

clf_svm= svm.SVC( kernel='rbf', gamma=10,probability=True)
clf_svm.fit(X_train,y_train)
#y_pred_svm = clf_svm.predict_proba(X_test)
y_pred_svm = clf_svm.predict(X_test)
#print(y_pred_svm)

#clf_rf = RandomForestClassifier(n_estimators=50, max_leaf_nodes=16, n_jobs=-1)
#clf_rf = RandomForestClassifier(n_estimators=70, max_depth=11, min_samples_split=50, min_samples_leaf=11)
clf_rf = RandomForestClassifier(n_estimators=90, max_depth=11, min_samples_split=80, min_samples_leaf=20)
clf_rf.fit(X_train,y_train)
#y_pred_rf = clf_rf.predict_proba(X_test)
y_pred_rf = clf_rf.predict(X_test)


#clf_ert = ExtraTreesClassifier(n_estimators=90, max_depth=None,
#                             min_samples_split=8, random_state=0)
clf_ert = ExtraTreesClassifier(n_estimators=90, max_depth=None,
                             min_samples_split=2, random_state=0)
clf_ert.fit(X_train,y_train)
#y_pred_ert = clf_ert.predict_proba(X_test)
y_pred_ert = clf_ert.predict(X_test)

clf_knn = KNeighborsClassifier()
clf_knn.fit(X_train,y_train)
y_pred_knn = clf_knn.predict(X_test)

clf_nb = GaussianNB()   # 使用默认配置初始化朴素贝叶斯
clf_nb.fit(X_train,y_train)    # 利用训练数据对模型参数进行估计
y_pred_nb = clf_nb.predict(X_test)     # 对参数进行预测
#clf_polyreg = PolynomialLogisticRegression(degree=20,C=0.1,penalty='l1')
#clf_polyreg.fit(X_train,y_train)
#y_pred_polyreg=clf_polyreg.predict(X_test)
#clf_ada=AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
#                         algorithm="SAMME",
#                         n_estimators=200, learning_rate=0.2)
#clf_ada=AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
#                         algorithm="SAMME",
#                         n_estimators=600, learning_rate=0.7)
clf_ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=1000, learning_rate=0.8)
clf_ada.fit(X_train, y_train)
y_pred_ada = clf_ada.predict(X_test)



#clf_gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
#clf_gp.fit(X_train, y_train)
#y_y_pred_gp,sigma2_pred=clf_gp.predict(X_test,eval_MSE=True)



clf_gp = GaussianProcessClassifier()
clf_gp.fit(X_train, y_train)
y_pred_gp=clf_gp.predict(X_test)


#clf_gb= GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=7,min_samples_leaf =60, min_samples_split =1200, max_features='sqrt',subsample=0.8, random_state=10)
#clf_gb.fit(X_train,y_train)
#y_pred_gb = clf_gb.predict(X_test)


#clf_log = LogisticRegression(random_state=42)
clf_log = LogisticRegression()
clf_log.fit(X_train,y_train)
y_pred_log = clf_log.predict(X_test)


#clf_xgb = XGBClassifier()
#clf_xgb.fit(X_train,y_train)
#y_pred_xgb = clf_xgb.predict(X_test)

#finalpred=(y_pred_svm+y_pred_rf+y_pred_ert)/3

#y_pred_ML=np.zeros((772,1))
#for i in range(3):
#    for j in range(2):
#        if((j+1)%2==0 and finalpred[i][j]>0.5):
 #           y_pred_ML[i]=1

#print(y_pred_ML.shape)
#y_pred_ML=y_pred_ML.reshape((1,772))













test_y=pd.read_table("/Users/yoland/PycharmProjects/untitled4/data/PNHT.te.label.txt",header=None)
predictions=np.loadtxt("/Users/yoland/PycharmProjects/untitled4/y_prob_mtx5.txt")
#svm_pred=np.loadtxt("/Users/yoland/PycharmProjects/untitled4/pred_svm_Matrix.txt")


#predictions=np.array(predictions)
print(predictions.shape)
pred_5=np.zeros((5,772))
for i in range(predictions.shape[0]):
    for j in range(predictions.shape[1]):
        if(predictions[i][j]>0.5):
            pred_5[i][j]=1
        else:
            pred_5[i][j]=0

#pred_5 = [np.mean(pred_5[:,col]) for col in range(np.size(pred_5, 1))]
#print(pred_5.shape)

#pred_svm=np.zeros((1,772))

test_y=(test_y+1)/2

#predictions = [np.mean(predictions[:,col]) for col in range(np.size(predictions, 1))]
#svm_pred=svm_pred.reshape((1,772))

#for i in range(svm_pred.shape[0]):
#    for j in range(svm_pred.shape[1]):
#        if(svm_pred[i][j]>0.5):
 #           pred_svm[i][j]=1
 #       else:
 #           pred_svm[i][j]=0
#
#predictions=pred_5
#predictions=np.vstack((pred_5,y_pred_svm))
#predictions=np.vstack((predictions,y_pred_rf))
#predictions=np.vstack((predictions,y_pred_ert))
predictions=np.vstack((y_pred_svm,y_pred_ert))
#predictions=np.vstack((predictions,y_pred_knn))
#predictions=np.vstack((predictions,y_pred_gb))
predictions=np.vstack((predictions,y_pred_ada))

#predictions=np.vstack((predictions,y_pred_nb))
#predictions=np.vstack((predictions,y_pred_gp))
predictions=np.vstack((predictions,y_pred_log))
#predictions=np.vstack((predictions,y_pred_polyreg))
#predictions=np.vstack((predictions,y_pred_xgb))

#predictions=np.vstack((pred_5,y_pred_ML))

#predictions = np.vstack((predictions,svm_pred))

print(test_y.shape)
print(predictions.shape)
print(len(predictions))
#predictions=np.transpose(predictions)
print(predictions.shape)
print(len(predictions))


def log_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
        final_prediction += weight * prediction

    return log_loss(test_y, final_prediction)



# the algorithms need a starting value, right not we chose 0.5 for all weights
# its better to choose many random starting points and run minimize a few times
#starting_values = [0.5] * len(predictions)
starting_values = [0.5] * len(predictions)
print(starting_values)
# adding constraints  and a different solver as suggested by user 16universe
# https://kaggle2.blob.core.windows.net/forum-message-attachments/75655/2393/otto%20model%20weights.pdf?sv=2012-02-12&se=2015-05-03T21%3A22%3A17Z&sr=b&sp=r&sig=rkeA7EJC%2BiQ%2FJ%2BcMpcA4lYQLFh6ubNqs2XAkGtFsAv0%3D
cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
# our weights are bound between 0 and 1
#bounds = [(0, 1)] * len(predictions)
bounds = [(0, 1)] * len(predictions)

res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)
#print(res.shape)
print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
print('Best Weights: {weights}'.format(weights=res['x']))
#print('Predictions: ' .format(predictions))
#print(np.array(predictions).shape)
print(res['x'])
matrix_prod=np.dot(res['x'],predictions)
#print(matrix_prod)
y_prob_ensemble= matrix_prod
y_pred_ensemble = [np.float(each > 0.5) for each in y_prob_ensemble]
y_prob_ensemble = np.array(y_prob_ensemble)
y_pred_ensemble = np.array(y_pred_ensemble)
print("y_prob_ensemble: ", y_prob_ensemble.shape)
print("y_pred_ensemble: ", y_pred_ensemble.shape)
print(test_y.shape)
y_true = test_y
print(y_true)
print(y_pred_ensemble)
auc_score_ensemble = metrics.roc_auc_score(y_true, y_prob_ensemble)
accuracy_score_ensemble = metrics.accuracy_score(y_true, y_pred_ensemble)

cm = metrics.confusion_matrix(y_true, y_pred_ensemble)


fpr,tpr,thresholds =metrics.roc_curve(y_true, y_prob_ensemble)
print(fpr.shape)
print(tpr.shape)
roc_auc=metrics.auc(fpr,tpr) #auc为Roc曲线下的面积

print(roc_auc)
#specificity_ensemble = cm[0, 0] / (cm[0, 0] + cm[0, 1])
#sensitivity_ensemble = cm[1, 1] / (cm[1, 1] + cm[1, 0])
print(cm)

print("Accuracy score (Testing Set) = ", accuracy_score_ensemble)
print("ROC AUC score  (Testing Set) = ", auc_score_ensemble)
#print("Sensitivity    (Testing Set) = ", sensitivity_ensemble)
#print("Specificity    (Testing Set) = ", specificity_ensemble)

#with open(testresult_fn, mode='a') as outfile:
#    outfile = csv.writer(outfile, delimiter=',')
#    outfile.writerow(
 #       ["ensemble", accuracy_score_ensemble, auc_score_ensemble, sensitivity_ensemble, specificity_ensemble])
plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
plt.legend(loc='lower right')
# plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate') #横坐标是fpr
plt.ylabel('True Positive Rate')  #纵坐标是tpr
plt.title('ROC curve')
plt.savefig(r'/Users/yoland/PycharmProjects/untitled4/AUC_compared.png')
plt.show()