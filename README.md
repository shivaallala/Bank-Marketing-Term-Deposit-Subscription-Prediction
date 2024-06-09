# Bank-Marketing-Term-Deposit-Subscription-Prediction

### Project Overview:

In this Project, our goal is to compare the performance of the classifiers we encountered in this section, namely K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines. We will utilize a dataset related to marketing bank products over the telephone.  

### Background:

Our dataset comes from the UCI Machine Learning repository [link](https://archive.ics.uci.edu/ml/datasets/bank+marketing).  The data is from a Portugese banking institution and is a collection of the results of multiple marketing campaigns.  We will make use of the article accompanying the dataset [here](CRISP-DM-BANK.pdf) for more information on the data and features.

The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).

### CRISP-DM Framework:

We will follow the CRISP-DM framework, a widely-used process in the industry for data projects:

#### Business Understanding

- **Objective:** Develop a predictive model to assist in targeted marketing efforts aimed at increasing subscription to term deposits. The primary focus of the model should be on accurately identifying customers who are likely to subscribe (i.e., those who have responded positively). While it's acceptable for the model to occasionally misclassify customers who are unlikely to subscribe as potential subscribers, it's crucial to minimize false negatives—instances where actual subscribers are incorrectly classified as non-subscribers. By prioritizing accuracy in identifying potential subscribers, the model aims to optimize resource allocation and maximize the effectiveness of marketing campaigns, ultimately enhancing the bank's subscription rate for term deposits.

#### Data Exploration

- **Initial dataset information**

![df info](./data/Images/df%20info.png)





##### Extreme outliers for price and odometer

![num features stats](/Images/Num_feat_stats.png)

##### Filling impurities:

1. **Year:** We can replace missing values with the most frequent year observed in the dataset.

