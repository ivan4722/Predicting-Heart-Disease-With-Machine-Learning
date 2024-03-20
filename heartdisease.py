import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc as sklearn_auc
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
data = pd.read_csv('heartdisease.csv')

X = data[['age', 'sex', 'chest pain type', 'resting bps', 'cholesterol', 'fasting blood sugar',
          'resting ecg', 'max heart rate', 'exercise angina', 'oldpeak', 'ST slope']]
y = data['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
'''
model = AdaBoostClassifier(random_state=42)

model.fit(X_train_scaled, y_train)

y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
fpr_original, tpr_original, thresholds_original = roc_curve(y_test, y_pred_proba)
roc_auc_original = sklearn_auc(fpr_original, tpr_original)
print("Original AUC:", roc_auc_original)
for col in X.columns:
    X_train_modified = X_train.drop(columns=[col])
    X_test_modified = X_test.drop(columns=[col])
    model.fit(X_train_modified, y_train)
    y_pred_proba_modified = model.predict_proba(X_test_modified)[:, 1]
    fpr_modified, tpr_modified, thresholds_modified = roc_curve(y_test, y_pred_proba_modified)
    roc_auc_modified = sklearn_auc(fpr_modified, tpr_modified)
    print(f"Dropping {col}: AUC = {roc_auc_modified}")
'''

# RANDOM FOREST 
model = RandomForestClassifier(random_state=42)

model.fit(X_train_scaled, y_train)

y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
fpr_original, tpr_original, thresholds_original = roc_curve(y_test, y_pred_proba)
roc_auc_original = sklearn_auc(fpr_original, tpr_original)

print("Original ROC AUC:", roc_auc_original)

for col in X.columns:
    X_train_modified = X_train.drop(columns=[col])
    X_test_modified = X_test.drop(columns=[col])
    
    model.fit(X_train_modified, y_train)
    
    y_pred_proba_modified = model.predict_proba(X_test_modified)[:, 1]
    fpr_modified, tpr_modified, thresholds_modified = roc_curve(y_test, y_pred_proba_modified)
    roc_auc_modified = sklearn_auc(fpr_modified, tpr_modified)
    
    print(f"Dropping {col}: ROC AUC = {roc_auc_modified}")


'''
# DECISION TREE
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
fpr_original, tpr_original, thresholds_original = roc_curve(y_test, y_pred_proba)
roc_auc_original = sklearn_auc(fpr_original, tpr_original)
print("Original AUC:", roc_auc_original)
for col in X.columns:
    X_train_modified = X_train.drop(columns=[col])
    X_test_modified = X_test.drop(columns=[col])
    model.fit(X_train_modified, y_train)
    y_pred_proba_modified = model.predict_proba(X_test_modified)[:, 1]
    fpr_modified, tpr_modified, thresholds_modified = roc_curve(y_test, y_pred_proba_modified)
    roc_auc_modified = sklearn_auc(fpr_modified, tpr_modified)
    print(f"Dropping {col}: AUC = {roc_auc_modified}")
'''
'''
# SVM
model = LinearSVC(dual=False)
model.fit(X_train_scaled, y_train)

y_pred_decision = model.decision_function(X_test_scaled)
fpr_original, tpr_original, thresholds_original = roc_curve(y_test, y_pred_decision)
roc_auc_original = sklearn_auc(fpr_original, tpr_original)

print("Original AUC:", roc_auc_original)

for col in X.columns:

    X_train_modified = X_train.drop(columns=[col])
    X_test_modified = X_test.drop(columns=[col])
    
    X_train_scaled_modified = scaler.fit_transform(X_train_modified)
    X_test_scaled_modified = scaler.transform(X_test_modified)
    
    model.fit(X_train_scaled_modified, y_train)

    y_pred_decision_modified = model.decision_function(X_test_scaled_modified)
    fpr_modified, tpr_modified, thresholds_modified = roc_curve(y_test, y_pred_decision_modified)
    roc_auc_modified = sklearn_auc(fpr_modified, tpr_modified)
    
    print(f"Dropping {col}: AUC = {roc_auc_modified}")
'''

'''
# LOGISTIC REGRESSION
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Calculate ROC AUC on the original dataset
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
fpr_original, tpr_original, thresholds_original = roc_curve(y_test, y_pred_proba)
roc_auc_original = sklearn_auc(fpr_original, tpr_original)

# Print original ROC AUC
print("Original AUC:", roc_auc_original)

# Iterate through each column, drop it, and calculate ROC AUC
for col in X.columns:
    # Drop the current column
    X_train_modified = X_train.drop(columns=[col])
    X_test_modified = X_test.drop(columns=[col])
    
    # Scale the modified features
    X_train_scaled_modified = scaler.fit_transform(X_train_modified)
    X_test_scaled_modified = scaler.transform(X_test_modified)
    
    # Fit the model on the modified dataset
    model.fit(X_train_scaled_modified, y_train)
    
    # Calculate ROC AUC on the modified dataset
    y_pred_proba_modified = model.predict_proba(X_test_scaled_modified)[:, 1]
    fpr_modified, tpr_modified, thresholds_modified = roc_curve(y_test, y_pred_proba_modified)
    roc_auc_modified = sklearn_auc(fpr_modified, tpr_modified)
    
    # Print the ROC AUC after dropping the current column
    print(f"Dropping {col}: AUC = {roc_auc_modified}")
'''