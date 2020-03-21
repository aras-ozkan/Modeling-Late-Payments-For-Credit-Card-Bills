import datetime
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def convert_to_learner_preds1(x_train):
    learner_preds = []
    tree_pred = tree1.predict(x_train)
    mlp_pred = mlp1.predict(x_train)
    learner_preds.append(tree_pred)
    learner_preds.append(mlp_pred)
    learner_preds = np.asarray(learner_preds).transpose()
    return learner_preds


def convert_to_learner_preds2(x_train):
    learner_preds = []
    tree_pred = tree2.predict(x_train)
    mlp_pred = mlp2.predict(x_train)
    learner_preds.append(tree_pred)
    learner_preds.append(mlp_pred)
    learner_preds = np.asarray(learner_preds).transpose()
    return learner_preds


def convert_to_learner_preds3(x_train):
    learner_preds = []
    tree_pred = tree3.predict(x_train)
    mlp_pred = mlp3.predict(x_train)
    learner_preds.append(tree_pred)
    learner_preds.append(mlp_pred)
    learner_preds = np.asarray(learner_preds).transpose()
    return learner_preds


def RMSE(preds, true):
    return np.sqrt(((preds - true) ** 2).sum() / preds.shape[0])


def eraseUnwanted(data):
    for c in data.columns:
        if(data[c].isnull().sum() > data.shape[0]*0.4):
            data.drop(columns=c, inplace=True)
    return data

def handleStrRows(data):
    strCols = data.select_dtypes('object')
    numCols = data.select_dtypes('int64', 'float64')
    imputerS = SimpleImputer(strategy="most_frequent").fit(strCols)
    imputer = SimpleImputer(strategy="mean").fit(numCols)
    strCols = imputerS.transform(strCols)
    numCols = imputer.transform(numCols)
    ohe = OneHotEncoder()
    enc = ohe.fit(strCols)
    strColsOhed = enc.transform(strCols).toarray()
    res = np.concatenate((numCols, strColsOhed), axis=1)
    return res


if __name__ == '__main__':

    # TARGET 1

    #   Import Data
    X1 = pd.read_csv('hw07_target1_training_data.csv', index_col='ID')
    y1 = pd.read_csv('hw07_target1_training_label.csv', index_col='ID').values
    X_t1 = pd.read_csv('hw07_target1_test_data.csv', index_col='ID')
    print('TARGET 1: ')
    train_num1 = X1.shape[0]
    X1 = pd.concat((X1, X_t1), axis=0)

    # Data Preprocess
    eraseUnwanted(X1)
    X1 = handleStrRows(X1)
    X_train_temp1 = X1[:train_num1]
    X_t1 = X1[train_num1:]
    print(X_t1.shape)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train_temp1, y1, test_size=0.2, stratify=y1)

    # Random Forest for learner 1
    tree1 = RandomForestClassifier(n_estimators=200, n_jobs=4)
    tree1 = tree1.fit(X_train1, y_train1.ravel())
    print("Random Forest Score: ", tree1.score(X_test1, y_test1))
    print(confusion_matrix(y_test1, tree1.predict(X_test1)))
    print('AUROC: ', roc_auc_score(y_test1, tree1.predict_proba(X_test1)[:, 1]))
    fpr, tpr, threshold = metrics.roc_curve(y_test1, tree1.predict_proba(X_test1)[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic for Random Forest 1')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # K-NN for learner 2
    mlp1 = KNeighborsClassifier(n_neighbors=5)
    mlp1 = mlp1.fit(X_train1, y_train1.ravel())
    print('K-Neighbours Classifier Score: ', mlp1.score(X_test1, y_test1))
    print(confusion_matrix(y_test1, mlp1.predict(X_test1)))
    print('AUROC: ', roc_auc_score(y_test1, mlp1.predict_proba(X_test1)[:, 1]))
    fpr, tpr, threshold = metrics.roc_curve(y_test1, mlp1.predict_proba(X_test1)[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic for K-Neighbours 1')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # Logistic Regression for ensemble
    GBReg1 = LogisticRegression().fit(convert_to_learner_preds1(X_train1), y_train1.ravel())
    print('LogisticRegression Score: ', GBReg1.score(convert_to_learner_preds1(X_test1), y_test1))

    # Confusion Matrix
    print(confusion_matrix(y_test1, GBReg1.predict(convert_to_learner_preds1(X_test1))))

    # AUROC
    print('AUROC: ', roc_auc_score(y_test1, GBReg1.predict_proba(convert_to_learner_preds1(X_test1))[:, 1]))
    fpr, tpr, threshold = metrics.roc_curve(y_test1, GBReg1.predict_proba(convert_to_learner_preds1(X_test1))[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic for Ensemble 1')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # Writing Test Results
    test_preds1 = tree1.predict_proba(X_t1)[:, 1]
    test_preds_df1 = pd.DataFrame(test_preds1)
    test_preds_df1.to_csv(header=False, index=False, path_or_buf='hw07_target1_test_predictions.csv')

    # TARGET 2

    #   Import Data
    X2 = pd.read_csv('hw07_target2_training_data.csv', index_col='ID')
    y2 = pd.read_csv('hw07_target2_training_label.csv', index_col='ID').values
    X_t2 = pd.read_csv('hw07_target2_test_data.csv', index_col='ID')
    print('TARGET 2: ')
    train_num2 = X2.shape[0]
    X2 = pd.concat((X2, X_t2), axis=0)

    # Data Preprocess
    eraseUnwanted(X2)
    X2 = handleStrRows(X2)
    X_train_temp2 = X2[:train_num2]
    X_t2 = X2[train_num2:]
    print(X_t2.shape)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train_temp2, y2, test_size=0.2, stratify=y2)

    # Random Forest for learner 1
    tree2 = RandomForestClassifier(n_estimators=200, n_jobs=4)
    tree2 = tree2.fit(X_train2, y_train2.ravel())
    print("Random Forest Score: ", tree2.score(X_test2, y_test2))
    print(confusion_matrix(y_test2, tree2.predict(X_test2)))
    print('AUROC: ', roc_auc_score(y_test2, tree2.predict_proba(X_test2)[:, 1]))
    fpr, tpr, threshold = metrics.roc_curve(y_test2, tree2.predict_proba(X_test2)[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic for Random Forest 2')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # K-NN for learner 2
    mlp2 = KNeighborsClassifier(n_neighbors=5)
    mlp2 = mlp2.fit(X_train2, y_train2.ravel())
    print('K-Neighbours Classifier Score: ', mlp2.score(X_test2, y_test2))
    print(confusion_matrix(y_test2, mlp2.predict(X_test2)))
    print('AUROC: ', roc_auc_score(y_test2, mlp2.predict_proba(X_test2)[:, 1]))
    fpr, tpr, threshold = metrics.roc_curve(y_test2, mlp2.predict_proba(X_test2)[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic for K-Neighbours 2')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # Logistic Regression for ensemble
    GBReg2 = LogisticRegression().fit(convert_to_learner_preds2(X_train2), y_train2.ravel())
    print('LogisticRegression Score: ', GBReg2.score(convert_to_learner_preds2(X_test2), y_test2))

    # Confusion Matrix
    print(confusion_matrix(y_test2, GBReg2.predict(convert_to_learner_preds2(X_test2))))

    # AUROC
    print('AUROC: ', roc_auc_score(y_test2, GBReg2.predict_proba(convert_to_learner_preds2(X_test2))[:, 1]))
    fpr, tpr, threshold = metrics.roc_curve(y_test2, GBReg2.predict_proba(convert_to_learner_preds2(X_test2))[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic for Ensemble 2')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # Writing Test Results
    test_preds2 = tree2.predict_proba(X_t2)[:, 1]
    test_preds_df2 = pd.DataFrame(test_preds2)
    test_preds_df2.to_csv(header=False, index=False, path_or_buf='hw07_target2_test_predictions.csv')

    # TARGET 3

    #   Import Data
    X3 = pd.read_csv('hw07_target3_training_data.csv', index_col='ID')
    y3 = pd.read_csv('hw07_target3_training_label.csv', index_col='ID').values
    X_t3 = pd.read_csv('hw07_target3_test_data.csv', index_col='ID')
    print('TARGET 3: ')
    train_num3 = X3.shape[0]
    X3 = pd.concat((X3, X_t3), axis=0)

    # Data Preprocess
    eraseUnwanted(X3)
    X3 = handleStrRows(X3)
    X_train_temp3 = X3[:train_num3]
    X_t3 = X3[train_num3:]
    print(X_t3.shape)
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X_train_temp3, y3, test_size=0.2, stratify=y3)

    # Random Forest for learner 1
    tree3 = RandomForestClassifier(n_estimators=200, n_jobs=4)
    tree3 = tree3.fit(X_train3, y_train3.ravel())
    print("Random Forest Score: ", tree3.score(X_test3, y_test3))
    print(confusion_matrix(y_test3, tree3.predict(X_test3)))
    print('AUROC: ', roc_auc_score(y_test3, tree3.predict_proba(X_test3)[:, 1]))
    fpr, tpr, threshold = metrics.roc_curve(y_test3, tree3.predict_proba(X_test3)[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic for Random Forest 3')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # K-NN for learner 2
    mlp3 = KNeighborsClassifier(n_neighbors=5)
    mlp3 = mlp3.fit(X_train3, y_train3.ravel())
    print('K-Neighbours Classifier Score: ', mlp3.score(X_test3, y_test3))
    print(confusion_matrix(y_test3, mlp3.predict(X_test3)))
    print('AUROC: ', roc_auc_score(y_test3, mlp3.predict_proba(X_test3)[:, 1]))
    fpr, tpr, threshold = metrics.roc_curve(y_test3, mlp3.predict_proba(X_test3)[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic for K-Neighbours 3')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # Logistic Regression for ensemble
    GBReg3 = LogisticRegression().fit(convert_to_learner_preds3(X_train3), y_train3.ravel())
    print('LogisticRegression Score: ', GBReg3.score(convert_to_learner_preds3(X_test3), y_test3))

    # Confusion Matrix
    print(confusion_matrix(y_test3, GBReg3.predict(convert_to_learner_preds3(X_test3))))

    # AUROC
    print('AUROC: ', roc_auc_score(y_test3, GBReg3.predict_proba(convert_to_learner_preds3(X_test3))[:, 1]))
    fpr, tpr, threshold = metrics.roc_curve(y_test3, GBReg3.predict_proba(convert_to_learner_preds3(X_test3))[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic for Ensemble 3')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # Writing Test Results
    test_preds3 = tree3.predict_proba(X_t3)[:, 1]
     hw08_pseudo_label.csv and data for it to hw08_pseudo_data.csv. I commented out pseudo label generation part. However, I added pseudo label generations results below. 


    Since I had only around 20,000 labeled datapoints for each target and my task is to predict 100,000, I had to increase the number of labeled datapoints. So, I created pseudo labels using this Random Forest. I created pseudo labels for all training data. But since this was not precise, I had to up sample correctly labeled data. I predicted other targets for each 20,000 datapoint (Each target has 20,000 correctly labeled datapoint so this gives us a 120,000 datapoints). And finally added the first predicted pseudo random of whole training data to it. I ended up with over 210,000 pseudo labeled datapoints. Then, I used these to train a new algorithm. However, since there are too many datapoints, my program takes a lot to execute. To speed up the process I saved my pseudo labels into csv file named 
