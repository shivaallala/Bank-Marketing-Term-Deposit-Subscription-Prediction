# Bank-Marketing-Term-Deposit-Subscription-Prediction

### Project Overview:

In this Project, our goal is to compare the performance of the classifiers we encountered in this section, namely K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines. We will utilize a dataset related to marketing bank products over the telephone.  

### Background:

Our dataset comes from the UCI Machine Learning repository [link](https://archive.ics.uci.edu/ml/datasets/bank+marketing).  The data is from a Portugese banking institution and is a collection of the results of multiple marketing campaigns.  We will make use of the article accompanying the dataset [here](CRISP-DM-BANK.pdf) for more information on the data and features.

The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).

#### CRISP-DM Framework:

We will follow the CRISP-DM framework, a widely-used process in the industry for data projects:

### Business Understanding

- **Objective:** Develop a predictive model to assist in targeted marketing efforts aimed at increasing subscription to term deposits. The primary focus of the model should be on accurately identifying customers who are likely to subscribe (i.e., those who have responded positively). While it's acceptable for the model to occasionally misclassify customers who are unlikely to subscribe as potential subscribers, it's crucial to minimize false negatives—instances where actual subscribers are incorrectly classified as non-subscribers. By prioritizing accuracy in identifying potential subscribers, the model aims to optimize resource allocation and maximize the effectiveness of marketing campaigns, ultimately enhancing the bank's subscription rate for term deposits.

### Data Exploration

- **Initial dataset information**

![df info](./data/Images/df%20info.png)


- **Initial dataset statistical summary for numerical features**

![stats summary](./data/Images/stats%20summary.png)

- **Understanding the Features**

- Input variables:

1. age (numeric)
2. job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3. marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4. education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5. default: has credit in default? (categorical: 'no','yes','unknown')
6. housing: has housing loan? (categorical: 'no','yes','unknown')
7. loan: has personal loan? (categorical: 'no','yes','unknown')

- related with the last contact of the current campaign:

8. contact: contact communication type (categorical: 'cellular','telephone')
9. month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10. day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11. duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

- other attributes:

12. campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13. pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14. previous: number of contacts performed before this campaign and for this client (numeric)
15.  poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

- social and economic context attributes

16. emp.var.rate: employment variation rate - quarterly indicator (numeric)
17. cons.price.idx: consumer price index - monthly indicator (numeric)
18. cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19. euribor3m: euribor 3 month rate - daily indicator (numeric)
20. nr.employed: number of employees - quarterly indicator (numeric)

- Output variable (desired target):

21. y - has the client subscribed a term deposit? (binary: 'yes','no')

We can see there are 11 categorical features and 10 numerical features. This dataset seems comprehensive and suitable for classification tasks. Before jumping into model building, it's crucial to understand the data. This involves exploring the distributions of features, handling outliers if any, dropping features that may not be useful and understanding the relationships between features and the target variable.


- The histograms for numerical features show the distribution of values for each numerical variable in the dataset. Each histogram helps to visualize the distribution of values for each numerical feature, which can provide insights into the underlying data distribution and potential outliers.

![Numerical Features Histogram](./data/Images/Numerical%20Features%20Histogram.png)

- Countplot of categorical features to visualize data distibution 

![Countplot of categorical features](./data/Images/Categorical%20Features%20countplot.png)


- Data Obervation

In the exploration of features containing unknown values, we find that several categorical features have a notable presence of 'unknown' values, including 'job', 'marital', 'education', 'default', 'housing', and 'loan'. These 'unknown' values could represent missing or unrecorded data, which need to be handled appropriately during preprocessing. Notably, 'default', 'housing', and 'loan' features have a significant number of 'unknown' values compared to their other categories. This observation suggests that these features may not be reliably recorded or may not be applicable to certain individuals. Additionally, the 'education' feature shows a range of education levels, with 'university.degree' being the most common followed by 'high.school' and 'basic.9y'. However, the presence of 'unknown' education levels underscores potential data quality issues. Furthermore, exploring correlations between features can provide insights into relationships that may exist among variables. For instance, correlations between features such as 'education' and 'job' could indicate associations between education levels and employment types. Overall, this initial exploration highlights the need for careful preprocessing, including handling unknown values and considering the relationships between features, to ensure the quality and effectiveness of subsequent modeling efforts.

- Correlation Matrix 

![Corr Matrix](./data/Images/correlation%20matrix.png)

The correlation matrix provides a visual representation of the relationships between numerical features in the dataset. In this heatmap, each cell represents the correlation coefficient between two features, ranging from -1 to 1. A correlation coefficient closer to 1 indicates a strong positive correlation, while a coefficient closer to -1 indicates a strong negative correlation. Features with a correlation coefficient close to 0 suggest a weak or no linear relationship.

From the heatmap, we can observe several noteworthy correlations:

**There is a strong positive correlation between 'euribor3m' and 'emp.var.rate', which is expected as these two features are related to economic indicators.**
**Similarly, 'nr.employed' and 'euribor3m' exhibit a strong positive correlation, indicating their association with employment rates and economic conditions.**
**Features such as 'cons.price.idx' and 'euribor3m' also show a moderate negative correlation, suggesting some inverse relationship between consumer price index and the three-month Euribor rate.**

Understanding these correlations can be valuable for feature selection and modeling. Highly correlated features may provide redundant information, which could potentially lead to overfitting in predictive models. Additionally, identifying correlated features can help in interpreting model results and understanding the underlying relationships in the data. However, it's important to note that correlation does not imply causation, and further analysis may be needed to validate any observed relationships.
