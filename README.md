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

In each of these techniques, rigorous analysis was done by plotting roc curves, learning rate curves , precision-recall curves and a detailed classification report.

# Model Evaluation
The above classifiers were evaluated based on how they performed on the test dataset.

Initially, the classifier was trained on a balanced training data set which was generated through undersampling by which best model parameters were chosen through GridSearch and Cross Validation technique

Then Receiver Operating characteristic(ROC) curve was plotted to check the model's performance and how much the model is overfitting the data.

Learning curve was plotted to visualise how model is being trained as more data has been added and to understand its biasing nature.

As a baseline , various metrics -->
  1. accuracy score
  2. precision score
  3. recall score
  4. ROC AUC score
  5. PR AUC score
  6. F1 score

were calculated with balanced training set and compared with imbalanced trainings set.

Finally, precision recall curve, confusion matrix and classification report was calculated on imbalanced testing dataset to fully understand the model's performance.

# Results and Discussion
It was seen that the dataset was highly imbalanced, it was necessary to examine and handle it carefully , which was done by undersampling technique.

All models tend to overfit the balanced dataset specially KNN and DTrees.

At the time of testing , it was seen that precision score was extremely less while recall score was really high. This suggests that model tend to identify most of the fraud cases which is a good sign, but it also falsely determine most of the non-fraud cases i.e it misclassified non-fraudulent transactions as fraudulent.

Although, it depends on company to company how to deal with it, usually identifing all the fraudulent cases is more of a priority than checking the non-fraud cases. But looking at the other side of the picture if non-frauds were classsified as frauds, users might stop transactioning through the company's card which would also degrade company's reputation.

Looking at both the outcomes, classifier's training conditions could be set based on what an organisation wants, example,high average precision,high recall,huigh precision, high ROC AUC,etc.

I have analysed the data through (neg_log_loss) scoring analysis on model, which generated different results, if you want a higher recall or higher precision in models prediction, you can set it accordingly.

# Conclusion
The project was successfully able to detect most of the fraudulent transactions which was unseen to the model and hopes to detect on more unseen data accurately as well.

This can be used by any government organisation or any private firm to keep in check how transactions are made and how cyber threats can be prevented.

# Dependencies and Installations

Following dependencies should be installed in your local machine -->

-Python 3.x

-Jupyter Notebook/Google Colab(Alternative)

-Numpy

-Pandas

-Matplotlib

-Seaborn

-sklearn

-imblearn

you can install the above by using the command

--> pip install (module)

# Directory Structure
CreditCard.ipynb -->Full code snippet with explanation

CreditCard_data.csv --> Dataset

LICENSE --> 'project licensed under MIT license'

# Contact Information
You can contact at -->
Email --> sankhlaaryan10@gmail.com

