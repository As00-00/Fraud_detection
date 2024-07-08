# Credit_Card Fraud Detection
This repository contains data analysis on credit card transaction to analyse and detect fraudulent transactions on Google Colab with the help of data analysis tools and classification machine learning models.
# Introduction
This is a very popular dataset openly available at Kaggle.

The major aim of this project is to help any organisation to be able to accurately and precisely detect the fraudulent transactions and to analyse the provided data itself and predict for future transactions as well.
# Dataset
The dataset to be analysed is a highly imbalanced data,for obvious reasons(Since,fraud transactions are extremely less likely to occur)

Dataset-->https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The data consists of around quarter a million observations i.e #transactions in the span of 2 days in September 2013 by European cardholders. Out  of which only 492 fraudulent transactions were recorded as per the data i.e only 0.17% (highly skewed data).

On record, there were plenty of original features of a particular transaction which included clients personal information, therefore for privacy purposes the data provided only had 31 features which were created by dimentionality reduction(PCA).

# Data Exploration and Visualisation

EDA(Exploratory data analysis) revealed that the dataset was highly skewed towards non-fraudulent transactions and highly imbalanced.

The average transaction amount was relatively low, <5000 euros (currency wasn't provided)

Higher amount transactions didn't mean fraud transactions as it was seen through scatter plots.

Majority of the predictors had Gausian Distribution of the data and outliers seemed to be very less.

Below link will redirect you to the directory where various plots and graphical analysis has been done to visualise the data and how the different training models have been trained through the data.

Visual directory-->https://github.com/As00-00/Fraud_detection/tree/main/EDA(Plots)

These were the final and most important plots, rest of the plots such as scatterplot,boxplots,etc you would find in code which were used for visualising the data in depth including outliers, sofisticated decision boundries and learning curves.

# Feature Engineering
There are 31 features which were already generated through Principle Componenty Analysis(PCA) of the original data , so there was no need to further reduce the dimentionality.

As of multicollinearity among the predictors, it was seen through correlation matrix that most of them were independent

TSNE(t-distributed stochastic neighbor embedding) was used to visualise the data in 2D , so as to further train the model effectively.

Apart from features, observations of majority class(non-frauds) were undersampled so that our model can learn the data by more precisely taking in account of minority class.

SMOTE(Synthetic Minority Oversampling Technique) was note used since the minority class was extremely rare to find (0.17%) and they would be high redundancy of the data.

# Modeling
At first, the data was scaled and the outliers were removed through Isolation Forest meathod.

Classifiers trained on the dataset were -->
  1. Logistic Regression
  2. K-Nearest Neighbors Classifier
  3. Support Vector Machines(SVM)
  4. Decision Trees



