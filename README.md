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

Below link will redirect you to the directory where various plots and graphical analysis has been done to visualise the data and how the different training models have been trained through the data.

Visual directory-->


