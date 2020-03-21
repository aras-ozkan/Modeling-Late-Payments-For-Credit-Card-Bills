# Modeling-Late-Payments-For-Credit-Card-Bills

In this project my task was to predict 3 different targets. They are all binary classification and have the same data fields. 
Data Preprocess:
The given data had many NaN values. I had to remove columns that have more than 40% NaN values with eraseUnwanted(data) method. After this I had to fill remaining NaN values in data. For number values that are missing, I used filling with mean. For String values, I used filling with most frequent. In order to have same features in test and train data, I added test data to train data as new data points and sent the joint data points through the same preprocess.

      Learners
	I used Random Forest, K-nn and LogisticRegression as learners. Most successful one was random forest in terms of AUROC. I used Logistic Regression to ensemble Random Forest and K-nn by using Random Forest and k-nn’s predictions as input data to train Logistic Regression. 

      AUROC
	I calculated each learners AUROC and score. I also drew the ROC curve for each one. In order to get a meaningful result I used probability of having a prediction value of 1 as AUROC input.


TARGET 1:
Processed Data Features: 131
Random Forest
Score:  0.8786363636363637
[[1798   74]
[ 193  135]]
AUROC:  0.9026621456118407
K-Neighbours Classifier
Score:  0.8322727272727273
[[1807   65]
[ 304   24]]
AUROC:  0.5360715355951636
LogisticRegression
Score:  0.8786363636363637
[[1798   74]
[ 193  135]]
AUROC:  0.6985054135397124

TARGET 2:
Processed Data Features: 108
Random Forest
Score:  0.945
[[1697    7]
[  92    4]]
AUROC:  0.7198717478482004
K-Neighbours Classifier
Score:  0.9416666666666667
[[1692   12]
[  93    3]]
AUROC:  0.6204457648669797
LogisticRegression
Score:  0.945
[[1697    7]
[  92    4]]
AUROC:  0.5263381504303599

TARGET 3:
Processed Data Features: 131
Random Forest Score:
0.863
[[854   5]
[132   9]]
AUROC:  0.8100380617409324
K-Neighbours Classifier
Score:  0.843
[[831  28]
[129  12]]
AUROC:  0.5799544249869962
LogisticRegression
Score:  0.863
[[854   5]
[132   9]]
AUROC:  0.5525474946127362
