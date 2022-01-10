# Air-Pressure-System-Failure-for-Scania-Trucks
Predict Failures and Minimize Costs based on Sensor Readings

Abstract

This report contains 2 sections. Section 1 is about predictive maintenance for air pressure system failure for Scania trucks to predict failures and minimize costs based on sensor readings while Logistic Regression model is built to predict type of failure and calculate the failure cost. Source code and data for section 1 can be view and retrieve from https://github.com/kelvinnghy96/Air-Pressure-System-Failure-for-Scania-Trucks.git. 
Section 2 is about stroke prediction in healthcare industry where 4 machine learning models approach which are Naïve Bayes, Logistic Regression, Random Forest, and Support Vector Machine (SVM) have been built to compare the accuracy among 4 models. Source code and data for section 2 can be view and retrieve from https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science.git. 

 
Section 1
Air Pressure System (APS) Failure for Scania Trucks

Air pressure system failure for Scania trucks is a predictive maintenance use case that predict the type of failure. Positive class failure consists of component failures for a specific component of the APS system while the negative class failure consists of truck with failures for components not related to the APS. Cost-metric of miss-classification is provided in the table below.
	True Class
	neg	pos
Predicted Class	neg	-	Cost_2 = 500
	pos	Cost_1 = 10	-
Table 1
	True Class
	neg	pos
Predicted Class	neg	TN	FP
	pos	FN	TP
Table 2
 
Figure 1
This use case has been used in an industrial challenge at year 2016 at the 15th International Symposium on Intelligent Data Analysis (IDA) event. In the past usage, top 3 scorer for minimum cost in that event is provided as reference in the table below.
Rank	Score	Number of Type 1 faults (FN)	Number of Type 2 faults (FP)
1	9920	542	9
2	10900	490	12
3	11480	398	15
Table 3
 
1.1	 Dataset Description

The dataset consists of data collected from heavy Scania trucks in everyday usage. The system in focus is the Air Pressure system (APS) which generates pressurized air that are utilized in various functions in a truck, such as braking and gear changes. The data consists of a subset of all available data, selected by experts. 
The dataset is split into train and test dataset and can be download from https://archive.ics.uci.edu/ml/machine-learning-databases/00421/. The training dataset contains 60000 examples in total in which 59000 belong to the negative class and 1000 for the positive class while test dataset contains 16000 examples. Both train and test dataset consist of 171 attributes. The attribute names of the data have been anonymized for proprietary reasons. The attributes are as class and other anonymized operational data.

1.2	Python Library

Library used in this assignment is list in the figure below.
 
Figure 2
 
1.3	Data Preprocessing

1.3.1	Missing Values
In the dataset, null values or missing values are denoted by "na". Missing values in the dataset is calculated in percentage and visualize in the figure below.
 
Figure 3
	A threshold of 30% is set as the percentage of missing values allowed in each attribute, therefore, attributes that contain more than 30% of missing values is removed from the train and test dataset. There are 10 attributes contain missing values more than 30% and is listed in the table below.
Attributes contain missing values more than 30%
br_000	bp_000	ab_000	bn_000	bl_000
bq_000	bo_000	cr_000	bm_000	bk_000
Table 4
	The leftover 161 attributes with missing values are filled up with their respective attribute’s median value. The reason of filling up missing value with median value instead of mean value is because some of the attributes contain outlier value which may contain important information and cant be removed from the dataset, therefore, median value is more suitable than mean value in this situation. 
1.3.2	Encoding Categorical Data
As Logistic Regression model is used in this assignment, all categorical data need to be encoded into numeric value. The only categorical attribute in the dataset is the ‘class’ attribute. Instead of using the LabelEncoder() function from sklearn library, an alternative way is used which direct replace the class label of ‘neg’ with 0 and ‘pos’ with 1 by using replace() function which provide the same output as LabelEncoder() function.

1.3.3	Feature Scaling with Standardization
Standardization is use in the feature scaling of this assignment. For every feature, all value gets rescale to mean value of 0 and a standard deviation of 1. The reason of using Standardization instead of Min-max Normalization is because if there’s outliers in feature, normalizing the data will scale most of the data to a small interval, which means all features will have the same scale but does not handle outliers well. Standardization is more robust to outliers, and in many cases, it is preferable over Min-max Normalization.

1.3.4	Class Balancing with Synthetic Minority Oversampling Technique (SMOTE)
SMOTE a statistical technique to generate new instances from existing minority cases to increase the number of cases in data in a balanced way. The target variable, ‘class’ contain imbalance data of ‘pos’ and ‘neg’ values which is 1,000 and 59,000 respectively while under SMOTE technique, ‘pos’ value is oversample to 59,000 rows of data to balance with the ‘neg’ value as shown in the figure below. Balanced data provide better accuracy and prevent model from penalizing minority samples.
Before SMOTE		After SMOTE
Category	Row Count		Category	Row Count
pos	1,000		pos	59,000
neg	59,000		neg	59,000
Figure 4 
1.3.5	Test Data Preprocessing
Apply the same preprocess model from train dataset to test dataset except SMOTE technique. SMOTE technique will only be applied in the train dataset. The data preprocessing steps applied in test dataset is list in below table.

No.	Test Data Preprocessing Steps
1	Drop column that contain more than 30% missing values
2	Replace missing values with median value
3	Encode class label from ‘neg’ to 0 and ‘pos’ to 1
4	Feature Scaling with Standardization
Table 5
 
1.4	Model Development

1.4.1	Logistic Regression
Logistic Regression model is used to predict type of failure in this assignment. Logistic Regression model is train before and after feature selection to compare the confusion matrix, model accuracy and the total cost.
False positive and false negative number in Logistic Regression model is dropped by 20% and 17.54% respectively after applied with feature selection as shown in the figure below.
Before Feature Selection		After Feature Selection
	True Class	Total			True Class	Total
	neg (0)	pos (1)				neg (0)	pos (1)	
Predicted Class	neg (0)	15,265	360	15,625
	Predicted Class	neg (0)	15,325	300	15,625

	pos (1)	57	318	375
		pos (1)	47	328	375

Total	15,322
678
16,000
	Total	15,372
628
16,000

Figure 5

Model accuracy is increase by 0.4375% after applied with feature selection as shown in the figure below.
Model Accuracy without Feature Selection	97.3937%
Model Accuracy with Feature Selection	97.8312%
Figure 6
 
Precision and recall is increase by 0.05 and 0.02 respectively after applied with feature selection while F1 Score is increase by 0.05 as shown in the figure below.
Before Feature Selection		After Feature Selection
Column	Precision	Recall	F1 Score		Column	Precision	Recall	F1 Score
neg (0)	1	0.98	0.99		neg (0)	1	0.98	0.99
pos (1)	0.47	0.85	0.6		pos (1)	0.52	0.87	0.65
Figure 7

Total cost is reduced by 30,100 as shown in the figure below. Although the cost of 150,470 is far away larger than the cost reference provided by past use case, but it still clearly shown that total cost is reduced after feature selection is applied.
Total Cost without Feature Selection	180,570
Total Cost with Feature Selection	150,470
Figure 8 
