# Table of Content
- [Air-Pressure-System-Failure-for-Scania-Trucks](#air-pressure-system-failure-for-scania-trucks)
- [Abstract](#abstract)
- [Air Pressure System (APS) Failure for Scania Trucks](#air-pressure-system-aps-failure-for-scania-trucks)
  - [1.1	 Dataset Description](#11-dataset-description)
  - [1.2	Python Library](#12python-library)
  - [1.3	Data Preprocessing](#13data-preprocessing)
    - [1.3.1	Missing Values](#131missing-values)
    - [1.3.2	Encoding Categorical Data](#132encoding-categorical-data)
    - [1.3.3	Feature Scaling with Standardization](#133feature-scaling-with-standardization)
    - [1.3.4	Class Balancing with Synthetic Minority Oversampling Technique (SMOTE)](#134class-balancing-with-synthetic-minority-oversampling-technique-smote)
    - [1.3.5	Test Data Preprocessing](#135test-data-preprocessing)
  - [1.4	Model Development](#14model-development)
    - [1.4.1	Logistic Regression](#141logistic-regression)
    - [1.4.2	Random Forest](#142random-forest)
    - [1.4.3	Summary](#143summary)

# Air-Pressure-System-Failure-for-Scania-Trucks
Predict Failures and Minimize Costs based on Sensor Readings

# Abstract

This repository is about predictive maintenance for air pressure system failure for Scania trucks to predict failures and minimize costs based on sensor readings while Logistic Regression model is built to predict type of failure and calculate the failure cost. Source code and data for this repository can be view and retrieve from https://github.com/kelvinnghy96/Air-Pressure-System-Failure-for-Scania-Trucks.git. 

# Air Pressure System (APS) Failure for Scania Trucks

Air pressure system failure for Scania trucks is a predictive maintenance use case that predict the type of failure. Positive class failure consists of component failures for a specific component of the APS system while the negative class failure consists of truck with failures for components not related to the APS. Cost-metric of miss-classification is provided in the table below.

![table1](https://github.com/kelvinnghy96/Air-Pressure-System-Failure-for-Scania-Trucks/blob/main/src/image/table1.png)

![table2](https://github.com/kelvinnghy96/Air-Pressure-System-Failure-for-Scania-Trucks/blob/main/src/image/table2.png)

This use case has been used in an industrial challenge at year 2016 at the 15th International Symposium on Intelligent Data Analysis (IDA) event. In the past usage, top 3 scorer for minimum cost in that event is provided as reference in the table below.

![figure1](https://github.com/kelvinnghy96/Air-Pressure-System-Failure-for-Scania-Trucks/blob/main/src/image/figure1.png)



## 1.1	 Dataset Description

The dataset consists of data collected from heavy Scania trucks in everyday usage. The system in focus is the Air Pressure system (APS) which generates pressurized air that are utilized in various functions in a truck, such as braking and gear changes. The data consists of a subset of all available data, selected by experts. 
The dataset is split into train and test dataset and can be download from https://archive.ics.uci.edu/ml/machine-learning-databases/00421/. The training dataset contains 60000 examples in total in which 59000 belong to the negative class and 1000 for the positive class while test dataset contains 16000 examples. Both train and test dataset consist of 171 attributes. The attribute names of the data have been anonymized for proprietary reasons. The attributes are as class and other anonymized operational data.

## 1.2	Python Library

Library used in this assignment is list in the figure below.
 
![figure2](https://github.com/kelvinnghy96/Air-Pressure-System-Failure-for-Scania-Trucks/blob/main/src/image/figure2.png)
 
## 1.3	Data Preprocessing

### 1.3.1	Missing Values
In the dataset, null values or missing values are denoted by "na". Missing values in the dataset is calculated in percentage and visualize in the figure below.
 
![figure3](https://github.com/kelvinnghy96/Air-Pressure-System-Failure-for-Scania-Trucks/blob/main/src/image/figure3.png)

A threshold of 30% is set as the percentage of missing values allowed in each attribute, therefore, attributes that contain more than 30% of missing values is removed from the train and test dataset. There are 10 attributes contain missing values more than 30% and is listed in the table below.

![table4](https://github.com/kelvinnghy96/Air-Pressure-System-Failure-for-Scania-Trucks/blob/main/src/image/table4.png)


The leftover 161 attributes with missing values are filled up with their respective attribute’s median value. The reason of filling up missing value with median value instead of mean value is because some of the attributes contain outlier value which may contain important information and cant be removed from the dataset, therefore, median value is more suitable than mean value in this situation. 
	
### 1.3.2	Encoding Categorical Data
As Logistic Regression model is used in this assignment, all categorical data need to be encoded into numeric value. The only categorical attribute in the dataset is the ‘class’ attribute. Instead of using the LabelEncoder() function from sklearn library, an alternative way is used which direct replace the class label of ‘neg’ with 0 and ‘pos’ with 1 by using replace() function which provide the same output as LabelEncoder() function.

### 1.3.3	Feature Scaling with Standardization
Standardization is use in the feature scaling of this assignment. For every feature, all value gets rescale to mean value of 0 and a standard deviation of 1. The reason of using Standardization instead of Min-max Normalization is because if there’s outliers in feature, normalizing the data will scale most of the data to a small interval, which means all features will have the same scale but does not handle outliers well. Standardization is more robust to outliers, and in many cases, it is preferable over Min-max Normalization.

### 1.3.4	Class Balancing with Synthetic Minority Oversampling Technique (SMOTE)
SMOTE a statistical technique to generate new instances from existing minority cases to increase the number of cases in data in a balanced way. The target variable, ‘class’ contain imbalance data of ‘pos’ and ‘neg’ values which is 1,000 and 59,000 respectively while under SMOTE technique, ‘pos’ value is oversample to 59,000 rows of data to balance with the ‘neg’ value as shown in the figure below. Balanced data provide better accuracy and prevent model from penalizing minority samples.

![figure4](https://github.com/kelvinnghy96/Air-Pressure-System-Failure-for-Scania-Trucks/blob/main/src/image/figure4.png)


### 1.3.5	Test Data Preprocessing
Apply the same preprocess model from train dataset to test dataset except SMOTE technique. SMOTE technique will only be applied in the train dataset. The data preprocessing steps applied in test dataset is list in below table.

![table5](https://github.com/kelvinnghy96/Air-Pressure-System-Failure-for-Scania-Trucks/blob/main/src/image/table5.png)

 
## 1.4	Model Development

### 1.4.1	Logistic Regression
Logistic Regression model is used to predict type of failure in this assignment. Logistic Regression model is train before and after feature selection to compare the confusion matrix, model accuracy and the total cost.
False positive and false negative number in Logistic Regression model is dropped by 20% and 17.54% respectively after applied with feature selection as shown in the figure below.

![figure5](https://github.com/kelvinnghy96/Air-Pressure-System-Failure-for-Scania-Trucks/blob/main/src/image/figure5.png)


Model accuracy is increase by 0.4375% after applied with feature selection as shown in the figure below.
![figure6](https://github.com/kelvinnghy96/Air-Pressure-System-Failure-for-Scania-Trucks/blob/main/src/image/figure6.png)

 
Precision and recall is increase by 0.05 and 0.02 respectively after applied with feature selection while F1 Score is increase by 0.05 as shown in the figure below.
![figure7](https://github.com/kelvinnghy96/Air-Pressure-System-Failure-for-Scania-Trucks/blob/main/src/image/figure7.png)


Total cost is reduced by 30,100 as shown in the figure below. Although the cost of 150,470 is far away larger than the cost reference provided by past use case, but it still clearly shown that total cost is reduced after feature selection is applied.
![figure8](https://github.com/kelvinnghy96/Air-Pressure-System-Failure-for-Scania-Trucks/blob/main/src/image/figure8.png)


### 1.4.2	Random Forest
Random Forest model is built with feature selection to compare the accuracy and total cost with Logistic Regression with feature selection. Confusion matrix for Random Forest model with feature selection is as below figure.

![figure9](https://github.com/kelvinnghy96/Air-Pressure-System-Failure-for-Scania-Trucks/blob/main/src/image/figure9.png)
 
Random Forest confusion matrix is compared with Logistic Regression confusion matrix. Although the FP is higher slightly by 2.33% but FN is lower by 29.79% as shown in figure below.
	
![figure10](https://github.com/kelvinnghy96/Air-Pressure-System-Failure-for-Scania-Trucks/blob/main/src/image/figure10.png)
 
Below figure will compare Precision, Recall and F1 Score between Random and Logistic Regression. Random Forest Precision, Recall and F1 Score is higher by 0.01, 0.04 and 0.02 respectively.

![figure11](https://github.com/kelvinnghy96/Air-Pressure-System-Failure-for-Scania-Trucks/blob/main/src/image/figure11.png)

Below figure will compare total cost and model accuracy between Random and Logistic Regression. Random Forest total cost is lower than Logistic Regression total cost by 6,930 while Random Forest model accuracy is higher than Logistic Regression model accuracy by 0.0438%

![figure12](https://github.com/kelvinnghy96/Air-Pressure-System-Failure-for-Scania-Trucks/blob/main/src/image/figure12.png)
![figure13](https://github.com/kelvinnghy96/Air-Pressure-System-Failure-for-Scania-Trucks/blob/main/src/image/figure13.png)

### 1.4.3	Summary
After perform feature selection the total cost of the model is reduced while the model accuracy increase. Logistic Regression with feature selection have a better model accuracy and total compare to Logistic Regression without feature selection but Random Forest with feature selection is the champion model in this assignment with a total cost of 19,570 and model accuracy of 97.8750%.

# License
![Click to view license][https://github.com/kelvinnghy96/Air-Pressure-System-Failure-for-Scania-Trucks/blob/67fa8295fac569ae07bc308ce1619634919e96c2/LICENSE]
