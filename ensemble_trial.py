import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn import metrics
import matplotlib.pylab as plt
from sklearn.ensemble import RandomForestClassifier
#from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.svm import SVC
from itertools import product
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import os
import random
from sklearn.model_selection import train_test_split
os.system("ls ../input")


X_train=pd.read_csv('/Users/yoland/PycharmProjects/untitled4/embed_HT_tr._1900.csv',header=None)
y_train=pd.read_table('/Users/yoland/PycharmProjects/untitled4/data/PNHT.tr.label.txt',header=None)
X_test=pd.read_csv('/Users/yoland/PycharmProjects/untitled4/embed_HT_te_1900.csv',header=None)


y_train=(y_train+1)/2


test_y=pd.read_table("/Users/yoland/PycharmProjects/untitled4/data/PNHT.te.label.txt",header=None)
predictions=np.loadtxt("/Users/yoland/PycharmProjects/untitled4/y_prob_mtx5.txt")
#svm_pred=np.loadtxt("/Users/yoland/PycharmProjects/untitled4/predMatrix.txt")


#svm_pred=svm_pred.reshape((1,772))

clf_svm= svm.SVC( kernel='rbf', gamma=10,probability=True)
clf_svm.fit(X_train,y_train)
#y_pred_svm = clf_svm.predict_proba(X_test)
y_pred_svm = clf_svm.predict(X_test)

clf_ert = ExtraTreesClassifier(n_estimators=90, max_depth=None,
                             min_samples_split=2, random_state=0)
clf_ert.fit(X_train,y_train)
#y_pred_ert = clf_ert.predict_proba(X_test)
y_pred_ert = clf_ert.predict(X_test)


clf_ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=1000, learning_rate=0.8)
clf_ada.fit(X_train, y_train)
y_pred_ada = clf_ada.predict(X_test)




clf_log = LogisticRegression()
clf_log.fit(X_train,y_train)
y_pred_log = clf_log.predict(X_test)


predictions_nb=np.vstack((y_pred_svm,y_pred_ert))
#predictions=np.vstack((predictions,y_pred_knn))
#predictions=np.vstack((predictions,y_pred_gb))
predictions_nb=np.vstack((predictions_nb,y_pred_ada))

#predictions=np.vstack((predictions,y_pred_nb))
#predictions=np.vstack((predictions,y_pred_gp))
predictions_nb=np.vstack((predictions_nb,y_pred_log))

def log_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions_nb):
        final_prediction += weight * prediction

    return log_loss(test_y, final_prediction)

starting_values = [0.5] * len(predictions_nb)
print(starting_values)
# adding constraints  and a different solver as suggested by user 16universe
# https://kaggle2.blob.core.windows.net/forum-message-attachments/75655/2393/otto%20model%20weights.pdf?sv=2012-02-12&se=2015-05-03T21%3A22%3A17Z&sr=b&sp=r&sig=rkeA7EJC%2BiQ%2FJ%2BcMpcA4lYQLFh6ubNqs2XAkGtFsAv0%3D
cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
# our weights are bound between 0 and 1
#bounds = [(0, 1)] * len(predictions)
bounds = [(0, 1)] * len(predictions_nb)

res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)
#print(res.shape)
print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
print('Best Weights: {weights}'.format(weights=res['x']))
#print('Predictions: ' .format(predictions))
#print(np.array(predictions).shape)
print(res['x'])
matrix_prod=np.dot(res['x'],predictions_nb)
#print(matrix_prod)
y_prob_nb= matrix_prod
y_prob_nb = np.array(y_prob_nb)










Ohp_Kmer_pred=predictions
Ohp_Kmer_pred = [np.mean(Ohp_Kmer_pred[:,col]) for col in range(np.size(Ohp_Kmer_pred, 1))]
Ohp_kmer_pred = [np.float(each > 0.5) for each in Ohp_Kmer_pred]



test_y=(test_y+1)/2

#predictions = [np.mean(predictions[:,col]) for col in range(np.size(predictions, 1))]


predictions = np.vstack((predictions,y_pred_svm))

print(test_y.shape)
print(predictions.shape)
print(len(predictions))
#predictions=np.transpose(predictions)
print(predictions.shape)
print(len(predictions))


predictions=np.vstack((predictions,y_pred_ert))
#predictions=np.vstack((predictions,y_pred_knn))
#predictions=np.vstack((predictions,y_pred_gb))
predictions=np.vstack((predictions,y_pred_ada))

#predictions=np.vstack((predictions,y_pred_nb))
#predictions=np.vstack((predictions,y_pred_gp))
predictions=np.vstack((predictions,y_pred_log))


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
print(y_pred_ensemble.shape)
auc_score_ensemble = metrics.roc_auc_score(y_true, y_prob_ensemble)
accuracy_score_ensemble = metrics.accuracy_score(y_true, y_pred_ensemble)

cm = metrics.confusion_matrix(y_true, y_pred_ensemble)


#svm_pred_score = [np.float(each > 0.5) for each in svm_pred.reshape((772,1))]
#svm_pred_score = svm_pred_score.astype(np.int)
#cm1 = metrics.confusion_matrix(y_true, svm_pred_score)

Ohp_Kmer_score=[np.float(each > 0.5) for each in Ohp_Kmer_pred]
cm2= metrics.confusion_matrix(y_true, Ohp_Kmer_score)



fpr1,tpr1,thresholds1 =metrics.roc_curve(y_true, Ohp_Kmer_pred)
fpr2,tpr2, thresholds2=metrics.roc_curve(y_true, y_prob_nb)
fpr,tpr,thresholds =metrics.roc_curve(y_true, y_prob_ensemble)
print(fpr1.shape)
print(tpr1.shape)
print(fpr.shape)
print(tpr.shape)
roc_auc=metrics.auc(fpr,tpr) #auc为Roc曲线下的面积
roc_auc1=metrics.auc(fpr1,tpr1)
roc_auc2=metrics.auc(fpr2,tpr2)
print(roc_auc)
print(roc_auc1)
print(roc_auc2)
#specificity_ensemble = cm[0, 0] / (cm[0, 0] + cm[0, 1])
#sensitivity_ensemble = cm[1, 1] / (cm[1, 1] + cm[1, 0])
#print(cm)
#print(cm1)
#print(cm2)

print("Accuracy score (Testing Set) = ", accuracy_score_ensemble)
print("ROC AUC score  (Testing Set) = ", auc_score_ensemble)
#print("Sensitivity    (Testing Set) = ", sensitivity_ensemble)
#print("Specificity    (Testing Set) = ", specificity_ensemble)

#with open(testresult_fn, mode='a') as outfile:
#    outfile = csv.writer(outfile, delimiter=',')
#    outfile.writerow(
labels = [ 'model 1 AUC = %0.3f'% roc_auc1 , 'model 2 AUC = %0.3f'% roc_auc2 , 'model 3 AUC = %0.3f'% roc_auc]
plt.figure()
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate') #the horizontal axis is fpr
plt.ylabel('True Positive Rate')  #the vertical axis is tpr
plt.title('ROC curve')
#auc_total = np.loadtxt('AUC_batch.txt')
#for i in range(np.shape(auc_total)[1]):
plt.plot(fpr1,tpr1,label=labels[0])
plt.plot(fpr2,tpr2,label=labels[1])
plt.plot(fpr,tpr, label= labels[2])

plt.legend()

 #       ["ensemble", accuracy_score_ensemble, auc_score_ensemble, sensitivity_ensemble, specificity_ensemble])
#plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
# plt.plot([0, 1], [0, 1], 'r--')

plt.savefig(r'/Users/yoland/PycharmProjects/untitled4/AUC_compared.png')
plt.show()
