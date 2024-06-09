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

#### Data distribution visualization 

- The histograms for numerical features show the distribution of values for each numerical variable in the dataset. Each histogram helps to visualize the distribution of values for each numerical feature, which can provide insights into the underlying data distribution and potential outliers.

![Numerical Features Histogram](./data/Images/Numerical%20Features%20Histogram.png)

- Countplot of categorical features to visualize data distibution 

![Countplot of categorical features](./data/Images/Categorical%20Features%20countplot.png)


#### Data Obervation

In the exploration of features containing unknown values, we find that several categorical features have a notable presence of 'unknown' values, including 'job', 'marital', 'education', 'default', 'housing', and 'loan'. These 'unknown' values could represent missing or unrecorded data, which need to be handled appropriately during preprocessing. Notably, 'default', 'housing', and 'loan' features have a significant number of 'unknown' values compared to their other categories. This observation suggests that these features may not be reliably recorded or may not be applicable to certain individuals. Additionally, the 'education' feature shows a range of education levels, with 'university.degree' being the most common followed by 'high.school' and 'basic.9y'. However, the presence of 'unknown' education levels underscores potential data quality issues. Furthermore, exploring correlations between features can provide insights into relationships that may exist among variables. For instance, correlations between features such as 'education' and 'job' could indicate associations between education levels and employment types. Overall, this initial exploration highlights the need for careful preprocessing, including handling unknown values and considering the relationships between features, to ensure the quality and effectiveness of subsequent modeling efforts.

#### Correlation Matrix 

![Corr Matrix](./data/Images/correlation%20matrix.png)

The correlation matrix provides a visual representation of the relationships between numerical features in the dataset. In this heatmap, each cell represents the correlation coefficient between two features, ranging from -1 to 1. A correlation coefficient closer to 1 indicates a strong positive correlation, while a coefficient closer to -1 indicates a strong negative correlation. Features with a correlation coefficient close to 0 suggest a weak or no linear relationship.

From the heatmap, we can observe several noteworthy correlations:

**There is a strong positive correlation between 'euribor3m' and 'emp.var.rate', which is expected as these two features are related to economic indicators.**
**Similarly, 'nr.employed' and 'euribor3m' exhibit a strong positive correlation, indicating their association with employment rates and economic conditions.**
**Features such as 'cons.price.idx' and 'euribor3m' also show a moderate negative correlation, suggesting some inverse relationship between consumer price index and the three-month Euribor rate.**

Understanding these correlations can be valuable for feature selection and modeling. Highly correlated features may provide redundant information, which could potentially lead to overfitting in predictive models. Additionally, identifying correlated features can help in interpreting model results and understanding the underlying relationships in the data. However, it's important to note that correlation does not imply causation, and further analysis may be needed to validate any observed relationships.


### Data Cleaning and Preprocessing

- Handling Missing Values and Unknown Categories:

For features like 'job', 'marital', 'education', 'housing', and 'loan', the 'unknown' values are addressed by replacing them with the mode of their respective columns. Since 'default' has a significant 'unknown' category and 'yes' is extremely rare, 'unknown' is treated as a separate category without any imputation. This ensures that the data is appropriately handled without losing information.

- Ordinal Encoding of Education:

The 'education' feature, which likely represents different levels of education, is ordinally encoded. This means assigning numerical values to education levels in a way that preserves their ordinal relationship. For example, 'basic.4y' might be encoded as 1, 'basic.6y' as 2, and so on. This encoding allows the model to understand the inherent order in education levels, which could be meaningful for analysis and prediction.

- Removing Extreme Outliers in Numerical Features:

Extreme outliers in numerical features are identified and removed to ensure the robustness and accuracy of the model. Outliers can significantly skew the distribution of data and affect the performance of machine learning algorithms. By removing these outliers, the dataset is better suited for modeling and analysis.

- Transforming Target Variable:

The target variable 'y', which likely represents whether a client has subscribed to a term deposit, is transformed from a categorical ('yes' and 'no') to a numerical (1 and 0) format. This transformation allows for easier analysis and modeling, as numerical representations are often more convenient for machine learning algorithms.

Overall, these preprocessing steps aim to ensure that the dataset is clean, properly formatted, and ready for analysis and modeling tasks. By addressing missing values, encoding categorical features, removing outliers, and transforming the target variable, the dataset becomes more suitable for predictive modeling and gaining insights into the underlying data patterns.


We are storing our clean data in new copied dataset called df_cleaned to track and seperate cleaned and original sets. 


#### Imputation treatment for Unknown values intuitively

Imputation treatment for unknown values intuitively addresses missing data in the dataset by making informed assumptions based on common sense or intuition.

- Imputation for 'job' and 'education':

The function impute_split_unknown is applied to the 'job' and 'education' features. This function identifies the top two modes (most frequent values) in each feature and then randomly assigns 'unknown' values approximately equally between these top two modes. This approach acknowledges that the 'unknown' values could reasonably belong to the most prevalent categories in the dataset. By splitting the 'unknown' values between the top two modes, it maintains the distribution of the original data while addressing missing values.

- Imputation for 'marital', 'housing', and 'loan':

For 'marital', the intuitive choice is to replace 'unknown' values with 'single'. This assumption is based on the idea that if marital status is unknown, assuming 'single' is a reasonable approximation.
For 'housing' and 'loan', the approach is to replace 'unknown' values with 'no'. This decision is grounded in the assumption that if information about housing or loan status is not available, it's more likely that the individual does not have a housing loan or a personal loan.

Overall, these imputation treatments aim to handle missing values in a pragmatic way that aligns with intuitive assumptions about the data. By making reasonable guesses based on the distribution of existing data, the imputation process ensures that missing values are filled in a manner that reflects the underlying patterns in the dataset.


#### Ordinal Encoding 'Education'

- Mapping Education Levels to Ordinal Values:

The education levels are mapped to ordinal values ranging from 0 to 6. The mapping is based on the provided edu_map dictionary, where each education level is associated with a numerical value. The levels are ordered based on their perceived hierarchy or level of education, with lower values representing lower levels of education and higher values representing higher levels.

![Education mapping](./data/Images/education%20mapping.png)

- Applying the Mapping:

The mapping is applied to the 'education' column of the DataFrame using the map function. This function replaces the original categorical education levels with their corresponding ordinal values according to the mapping defined in edu_map.

- Result:

After applying the ordinal encoding, the 'education' column now contains numerical values representing the education levels instead of the original categorical labels. This transformation allows machine learning algorithms to interpret the education levels as ordered categories, which can potentially improve the model's performance by capturing the inherent ordinal relationship between education levels.

Overall, the code snippet demonstrates a straightforward way to encode categorical variables with an ordinal relationship into numerical values, making them suitable for analysis and modeling purposes.


#### New feature pre_contact from pdays

By analyzing the distribution of the 'pdays' feature, which represents the number of days since the client was last contacted from a previous campaign. The percentage of clients not contacted prior to the current campaign is calculated and displayed, showing that approximately 96.32% of clients fall into this category. The subsequent plot visually illustrates this, with the count of clients having 'pdays' equal to 999 being significantly higher than all other unique values combined. Recognizing the imbalance and the limited insight provided by the actual number of days, a decision is made to transform the feature. A new binary feature called 'pre_contact' is created, indicating whether a client has been previously contacted or not. The original 'pdays' feature is then dropped from the dataset to streamline it. This transformation simplifies the representation of contact history, focusing on the presence or absence of previous contact rather than specific time intervals, which may not provide significant predictive value due to the dominance of '999' values.

![pdays distribution](./data/Images/pdays%20distribution.png)

From the plot above, it is clear that the value '999' in pdays attribute is significantly higher than all the other unique values put together. This means about 96.32% of the clients were not contacted prior to this campaign resulting high skewed data for 'pdays'. I believe we can tranform this feature by creating a new feature that exhibits whether a client has been previously contacted or not. The quantity of previous contact poses no real insight due to lack of data in comparison to the value of the 'pdays' features that represents clients have not been contact previously.

A new feature 'pre_contact' will be created from 'pday' and then 'pdays' will be dropped.

