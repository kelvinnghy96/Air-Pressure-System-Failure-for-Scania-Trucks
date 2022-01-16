# Air Pressure System (APS) Failure for Scania Trucks  
## Predict Failures and Minimize Costs based on Sensor Readings

# 1. DATA PREPROCESSING

## DOWNLOAD DATA
### Download train data and test data from https://archive.ics.uci.edu/ml/machine-learning-databases/00421/ and store those downloaded .csv file in the 'data' folder

## IMPORT LIBRARY
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import scikitplot as skplt
from imblearn.over_sampling import SMOTE


## SET SEED
np.random.seed(2022)


## IMPORT DATA  
### Import train data and test data from .csv file downloaded.
### Train Data
train_data = pd.read_csv('../data/aps_failure_training_set.csv',skiprows=20,na_values="na")
train_data.head()

###Test Data
test_data = pd.read_csv('../data/aps_failure_test_set.csv',skiprows=20,na_values="na")
test_data.head()


## TRAIN DATA PREPROCESSING
## MISSING VALUE
fig, ax = plt.subplots(figsize=(15,5))
threshold = 30

### Calculate % of missing values for each attribute
missing = train_data.isna().sum().div(train_data.shape[0]).mul(100).to_frame().sort_values(by=0,ascending = False )
ax.bar(missing.index, missing.values.T[0])
plt.title("Percentage of Missing Values in each Attribute")
plt.xticks([])
plt.xlabel("Attributes")
plt.ylabel("Percentage missing")
plt.axhline(threshold, color='red', ls='dotted')
plt.show()


### Count amount of columns that contain missing values more than 30%.
### Store index of columns that contain missing values more than threshold.
cols_missing_30 = missing[missing[0]>30].index
no_cols_na = len(cols_missing_30)
### Count and print amount of column more than threshold.
print("There are {0} columns contain missing values more than {1}%.".format(no_cols_na,threshold))
cols_missing_30


## DROP COLUMNS  
### Drop columns that contain missing values more than 30%.
### Drop columns that contain missing values more than 30%.
train_data_drp30 = train_data.drop(cols_missing_30, axis=1)

### Validate only selected column is removed.
train_data_drp30.shape


## REPLACE MISSING VALUES  
### Replace leftover missing values with median.
train_data_fillna = train_data_drp30.fillna(train_data_drp30.median(), inplace=False)

### Validate no missing values left
train_data_fillna.isnull().sum().sum()


## ENCODING CATEGORICAL DATA  
### Encode categorical data into numeric type data.
### Check categorical column in data
train_data_fillna.info()
train_data_fillna['class'].value_counts()

### Replace class label from (neg,pos) to (0,1)
train_data_encode = train_data_fillna.replace('neg',0).replace('pos',1)

### Validate no categorical data left.
train_data_encode.info()


## SPLIT TARGET VARIABLE IN TRAIN DATA  
### Split train data to x_train and y_train
x_train = train_data_encode.drop('class', axis=1)
x_train.head()
y_train = train_data_encode.iloc[:,0]
y_train.head()
y_train.value_counts()


## FEATURE SCALING WITH STANDARDIZATION  
### For every feature, all value gets rescale to mean value of 0 and a standard deviation of 1.
### Define standardization
scaler = StandardScaler()

### Data rescaled to mean value of 0 and a standard deviation of 1
x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train))
x_train_scaled

### Validate data is rescaled
df_mean = x_train_scaled.mean().sum()
df_std = x_train_scaled.std().round().max()
print("The mean of each attribute in the data is", round(df_mean))
print("The standard deviation of each attribute in the data is", round(df_std))


## CLASS BALANCING WITH SMOTE  
### Synthetic Minority Oversampling Technique (SMOTE).  
### Generating new instances from existing minority cases to increase the number of cases in data in a balanced way. 

## SMOTE
sm = SMOTE()
x_train_res, y_train_res = sm.fit_resample(x_train_scaled, y_train)

### Validate data is resampled
y_train_res.value_counts()


# TEST DATA PREPROCESSING
## APPLY PREPROCESS MODEL  
### Apply same preprocess model from train data to test data.
### Drop columns that contain missing values more than 30%.
test_data_drp30 = test_data.drop(cols_missing_30, axis=1)

### Replace leftover missing values with median.
test_data_fillna = test_data_drp30.fillna(test_data_drp30.median(), inplace=False)

### Replace class label from (neg,pos) to (0,1)
test_data_encode = test_data_fillna.replace('neg',0).replace('pos',1)


## SPLIT TARGET VARIABLE IN TEST DATA  
### Split test data to x_test and y_test
x_test = test_data_encode.drop('class', axis=1)
x_test.head()
y_test = test_data_encode.iloc[:,0]
y_test.head()

### Data rescaled to mean value of 0 and a standard deviation of 1
x_test_scaled = pd.DataFrame(scaler.fit_transform(x_test))
x_train_final = x_train_res
y_train_final = y_train_res
x_test_final = x_test_scaled
y_test_final = y_test


# 2. MODEL DEVELOPMENT - LOGISTIC REGRESSION

## LOGISTIC REGRESSION MODEL ACCURACY BEFORE FEATURE SELECTION
model = LogisticRegression(max_iter=1000)
model.fit(x_train_final, y_train_final)
model_score = model.score(x_test_final, y_test_final) * 100
print("Model Accuracy without Feature Selection: {:.4f}%".format(model_score))

## CONFUSION MATRIX BEFORE FEATURE SELECTION
pred = model.predict(x_test_final)
tn, fp, fn, tp = confusion_matrix(y_test_final, pred).ravel()
skplt.metrics.plot_confusion_matrix(y_test_final, pred, normalize=False)
plt.show()

## CLASSFICATION REPORT BEFORE FEATURE SELECTION
### Classification Report
print(classification_report(y_test_final, pred))


## TOTAL COST BEFORE FEATURE SELECTION
### Total Cost before Feature Selection
### Good condition but predicted as faulty, cost of 10 for maintainance fee
cost_1 = 10

### Faulty but predicted as good condition, cost of 500 for maintainance fee
cost_2 = 500

cost = fp * cost_1 + fn * cost_2
cost


## FEATURE SELECTION
### Selecting the Best important features according to Logistic Regression using SelectFromModel
sfm_logreg = SelectFromModel(estimator=LogisticRegression(max_iter=1000))
sfm_logreg.fit(x_train_final, y_train_final)
feature_selected_tmp = x_train_final.columns[sfm_logreg.get_support()]
feature_selected_tmp
feature_selected = feature_selected_tmp
x_train_final_fs = x_train_final.iloc[:,feature_selected]
x_train_final_fs.head()
x_test_final_fs = x_test_final.iloc[:,feature_selected]
x_test_final_fs.head()


## LOGISTIC REGRESSION MODEL ACCURACY AFTER FEATURE SELECTION
model_fs = LogisticRegression(max_iter=1000)
model_fs.fit(x_train_final_fs, y_train_final)
model_fs_score = model_fs.score(x_test_final_fs, y_test_final) * 100

print("Model Accuracy without Feature Selection: {:.4f}%".format(model_score))
print("Model Accuracy with Feature Selection: {:.4f}%".format(model_fs_score))


## CONFUSION MATRIX AFTER FEATURE SELECTION
pred_fs = model_fs.predict(x_test_final_fs)
tn, fp, fn, tp = confusion_matrix(y_test_final, pred_fs).ravel()
skplt.metrics.plot_confusion_matrix(y_test_final, pred_fs, normalize=False)
plt.show()


## CLASSIFICATION REPORT AFTER FEATURE SELECTION
### Classification Report after feature selection
print(classification_report(y_test_final, pred_fs))


## TOTAL COST AFTER FEATURE SELECTION
### Total Cost after Feature Selection
### Good condition but predicted as faulty, cost of 10 for maintainance fee
cost_1 = 10

### Faulty but predicted as good condition, cost of 500 for maintainance fee
cost_2 = 500

cost_fs = fp * cost_1 + fn * cost_2
cost_fs


## LOGISTICS REGRESSION SUMMARY
print("Model Accuracy without Feature Selection: {:.4f}%".format(model_score))
print("Model Accuracy with Feature Selection: {:.4f}%".format(model_fs_score))

print("Total Cost without Feature Selection: {:.0f}".format(cost))
print("Total Cost with Feature Selection: {:.0f}".format(cost_fs))

# 3. MODEL DEVELOPMENT - RANDOM FOREST
## RANDOM FOREST MODEL ACCURACY WITH FEATURE SELECTION
model_rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1)
model_rf.fit(x_train_final_fs, y_train_final)
model_rf_score = model_rf.score(x_test_final_fs, y_test_final) * 100
print("Random Forest Model Accuracy with Feature Selection: {:.4f}%".format(model_rf_score))


## RANDOM FOREST MODEL CONFUSION MATRIX
pred_rf = model_rf.predict(x_test_final_fs)
tn, fp, fn, tp = confusion_matrix(y_test_final, pred_rf).ravel()
skplt.metrics.plot_confusion_matrix(y_test_final, pred_rf, normalize=False)
plt.show()


## RANDOM FOREST MODEL TOTAL COST
cost_rf = fp * cost_1 + fn * cost_2
print("Random Forest Model Total Cost with Feature Selection: {:.0f}".format(cost_rf))


## RANDOM FOREST CLASSFICATION REPORT WITH FEATURE SELECTION
### Random Forest Classification Report
print(classification_report(y_test_final, pred_rf))


## RANDOM FOREST SUMMARY
print("Random Forest Model Accuracy with Feature Selection: {:.4f}%".format(model_rf_score))
print("Random Forest Model Total Cost with Feature Selection: {:.0f}".format(cost_rf))

