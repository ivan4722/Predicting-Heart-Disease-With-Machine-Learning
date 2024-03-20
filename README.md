# Predicting Heart Disease Using Machine Learning
dataset: https://www.kaggle.com/datasets/winson13/heart-disease-dataset?resource=download

We can build models using the following:
- logistic regression
- SVM
- decision tree
- random forest
- adaBoost

We can then calculate the accuracy of our model using the AUC (area under the curve) of the ROC curve. The ROC curve compares the true positive to the false negative rate. That way, we can ensure sensitivity and specificity of our model.

Below are the AUC of each model:
```
Logistic regression: 0.82138969521045
SVM: 0.82138969521045
Decision tree: 0.6760703918722786
Random forest: 0.8599873004354137
adaBoost: 0.8102322206095791
```
The AUC scores suggest that random forest is the best classifier, while decision tree is the worst. 

# What is the best predictor? 
The dataset contains the following predictors:\
age,sex,chest pain type,resting bps,cholesterol,fasting blood sugar,resting ecg,max heart rate,exercise angina,oldpeak,ST slope

We can compute the AUC score dropping each predictor one by one. Whichever one drops the AUC score the most has the greatest impact on the AUC, 
and this we can choose that as the best predictor. 
```
Logistic Regression
Original AUC: 0.82138969521045
Dropping age: AUC = 0.8309143686502177
Dropping sex: AUC = 0.8188497822931785
Dropping chest pain type: AUC = 0.8037917271407837
Dropping resting bps: AUC = 0.8214804063860668
Dropping cholesterol: AUC = 0.8213896952104499
Dropping fasting blood sugar: AUC = 0.8203918722786647
Dropping resting ecg: AUC = 0.8195754716981132
Dropping max heart rate: AUC = 0.8290094339622642
Dropping exercise angina: AUC = 0.8195754716981132
Dropping oldpeak: AUC = 0.8183962264150942
Dropping ST slope: AUC = 0.7760341074020319
SVM
Original AUC: 0.82138969521045
Dropping age: AUC = 0.830188679245283
Dropping sex: AUC = 0.8195754716981132
Dropping chest pain type: AUC = 0.8051523947750363
Dropping resting bps: AUC = 0.8214804063860668
Dropping cholesterol: AUC = 0.8216618287373004
Dropping fasting blood sugar: AUC = 0.8212082728592163
Dropping resting ecg: AUC = 0.8202104499274312
Dropping max heart rate: AUC = 0.8284651669085631
Dropping exercise angina: AUC = 0.8204825834542816
Dropping oldpeak: AUC = 0.81966618287373
Dropping ST slope: AUC = 0.7764876632801161
Decision Tree
Original AUC: 0.6760703918722786
Dropping age: AUC = 0.6949383164005805
Dropping sex: AUC = 0.6996552975326561
Dropping chest pain type: AUC = 0.7187953555878084
Dropping resting bps: AUC = 0.7280478955007258
Dropping cholesterol: AUC = 0.67144412191582
Dropping fasting blood sugar: AUC = 0.6759796806966618
Dropping resting ecg: AUC = 0.7045537010159652
Dropping max heart rate: AUC = 0.6762518142235123
Dropping exercise angina: AUC = 0.6806966618287373
Dropping oldpeak: AUC = 0.6806966618287373
Dropping ST slope: AUC = 0.6859579100145139
Random Forest
Original ROC AUC: 0.8599873004354137
Dropping age: ROC AUC = 0.8696026850507982
Dropping sex: ROC AUC = 0.8518686502177069
Dropping chest pain type: ROC AUC = 0.8118196661828737
Dropping resting bps: ROC AUC = 0.8574020319303338
Dropping cholesterol: ROC AUC = 0.8521407837445574
Dropping fasting blood sugar: ROC AUC = 0.8586719883889695
Dropping resting ecg: ROC AUC = 0.8413007982583455
Dropping max heart rate: ROC AUC = 0.8575380986937591
Dropping exercise angina: ROC AUC = 0.8495555152394774
Dropping oldpeak: ROC AUC = 0.8509615384615383
Dropping ST slope: ROC AUC = 0.7580279390420901
adaBoost
Original AUC: 0.8102322206095791
Dropping age: AUC = 0.8163098693759071
Dropping sex: AUC = 0.8051977503628447
Dropping chest pain type: AUC = 0.7567579825834542
Dropping resting bps: AUC = 0.8147224238026124
Dropping cholesterol: AUC = 0.8236574746008709
Dropping fasting blood sugar: AUC = 0.8140874455732946
Dropping resting ecg: AUC = 0.7873730043541364
Dropping max heart rate: AUC = 0.8122732220609579
Dropping exercise angina: AUC = 0.8102322206095791
Dropping oldpeak: AUC = 0.8065584179970972
Dropping ST slope: AUC = 0.7216981132075471
```
We can see that in logistic regression, SVM, decision tree, and random forest dropping ST slope has the greatest impact on the AUC.
However, in decision tree, the original AUC is actually the lowest. This suggests that in the context of predicting heart disease,
ST slope as a predictor has strong discrimanatory ability. However, in most models tested (4/5), the AUC of the original model is the greatest,
demonstrating the importance of considering all factors/features.
