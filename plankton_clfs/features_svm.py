import numpy as np
import gzip
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def load_data():
    train_path = "./laps_nobg_100/features_nounk_079.npy"
    labels_path = "./laps_nobg_100/labels_train.npy.gz"
    with gzip.open(labels_path, "rb") as f:
        labels = np.load(f)

    train_idx = np.load("./laps_nobg_100/indices_train.npy")
    valid_idx = np.load("./laps_nobg_100/indices_valid.npy")

    feat = np.load(train_path)
    
    X_train, y_train = feat[train_idx], labels[train_idx]
    X_valid, y_valid = feat[valid_idx], labels[valid_idx]

    return X_train, y_train, X_valid, y_valid

X_train, y_train, X_valid, y_valid = load_data()

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5 )
clf.fit(X_train, y_train)
print("Best parameters set found on development set:")
print(clf.best_params_)
print("Grid scores on development set:")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))

print("Detailed classification report:")
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
y_true, y_pred = y_valid, clf.predict(X_valid)
print(classification_report(y_true, y_pred))

np.save("./laps_nobg_100/predictions_transfer_079_nounk.npy", y_pred)
np.save("./laps_nobg_100/real_labels_transfer_079_nounk.npy", y_true)

print("Predictions saved.")

#clf = linear_model.LogisticRegression()
#
#clf.fit(X_train, y_train)
#score = clf.score(X_valid, y_valid)
#print(score)
#
#from sklearn import svm
#
#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
#clfs = []
#
#parameters = [['svc','linear', '1'], ['svc', 'linear', '10'], ['svc', 'rbf',\
#                '1'], ['svc', 'rbf', '10'], ['svc', 'rbf', '20'], ['linearsvc', '1', '100k', 'primal'],\
#                ['linearsvc', '10', '100k', 'primal'], ['linearsvc', '10', '100k', 'dual'], \
#                ['linearsvc', '20', '100k', 'primal']]
#
#clfs.append(svm.SVC(kernel='linear', C=1, max_iter=10000))
#clfs.append(svm.SVC(kernel='linear', C=10, max_iter=10000))
#clfs.append(svm.SVC(kernel='rbf', C=1, max_iter=10000))
#clfs.append(svm.SVC(kernel='rbf', C=5, max_iter=100000))
#clfs.append(svm.SVC(kernel='rbf', C=10, max_iter=100000))
#clfs.append(svm.LinearSVC(C=1, max_iter=100000, dual=False))
#clfs.append(svm.LinearSVC(C=10, max_iter=100000, dual=False))
#clfs.append(svm.LinearSVC(C=10, max_iter=100000, dual=True))
#clfs.append(svm.LinearSVC(C=20, max_iter=100000, dual=False))
#
#for i in range(9):
#    print(parameters[i])
#    clfs[i].fit(X_train, y_train)
#    print(clfs[i].score(X_valid, y_valid))

#svm_clf = svm.LinearSVC()
#svm_clf.fit(X_train, y_train) 
#print(svm_clf.score(X_valid, y_valid))
