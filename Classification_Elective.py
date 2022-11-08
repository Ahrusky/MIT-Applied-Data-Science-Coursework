#!/usr/bin/env python
# coding: utf-8

# 
# 
# 
# # **Decision Trees and Random Forest Project: Predicting Potential Customers**
# 
# # **Marks: 60**

# Welcome to the project on classification using Decision Tree and Random Forest.
# 
# --------------------------------
# ## **Context** 
# -------------------------------
# 
# The EdTech industry has been surging in the past decade immensely, and according to a forecast, the Online Education market, would be worth $286.62bn by 2023, with a compound annual growth rate (CAGR) of 10.26% from 2018 to 2023. The modern era of online education has enforced a lot in its growth and expansion beyond any limit. Due to having many dominant features like ease of information sharing, personalized learning experience, transparency of assessment, etc., it is now preferable to traditional education. 
# 
# The online education sector has witnessed rapid growth and is attracting a lot of new customers. Due to this rapid growth, many new companies have emerged in this industry. With the availability and ease of use of digital marketing resources, companies can reach out to a wider audience with their offerings. The customers who show interest in these offerings are termed as **leads**. There are various sources of obtaining leads for Edtech companies, like:
# 
# * The customer interacts with the marketing front on social media or other online platforms. 
# * The customer browses the website/app and downloads the brochure.
# * The customer connects through emails for more information.
# 
# The company then nurtures these leads and tries to convert them to paid customers. For this, the representative from the organization connects with the lead on call or through email to share further details.
# 
# 
# ----------------------------
# ## **Objective**
# -----------------------------
# 
# 
# ExtraaLearn is an initial stage startup that offers programs on cutting-edge technologies to students and professionals to help them upskill/reskill. With a large number of leads being generated on a regular basis, one of the issues faced by ExtraaLearn is to identify which of the leads are more likely to convert so that they can allocate the resources accordingly. You, as a data scientist at ExtraaLearn, have been provided the leads data to:
# * Analyze and build an ML model to help identify which leads are more likely to convert to paid customers. 
# * Find the factors driving the lead conversion process.
# * Create a profile of the leads which are likely to convert.
# 
# 
# --------------------------
# ## **Data Description**
# --------------------------
# 
# The data contains the different attributes of leads and their interaction details with ExtraaLearn. The detailed data dictionary is given below.
# 
# * **ID:** ID of the lead
# * **age:** Age of the lead
# * **current_occupation:** Current occupation of the lead. Values include 'Professional', 'Unemployed', and 'Student'
# * **first_interaction:** How did the lead first interact with ExtraaLearn? Values include 'Website' and 'Mobile App'
# * **profile_completed:** What percentage of the profile has been filled by the lead on the website/mobile app? Values include Low - (0-50%), Medium - (50-75%), High (75-100%)
# * **website_visits:** The number of times a lead has visited the website
# * **time_spent_on_website:** Total time (seconds) spent on the website.
# * **page_views_per_visit:** Average number of pages on the website viewed during the visits
# * **last_activity:** Last interaction between the lead and ExtraaLearn 
#     * **Email Activity:** Seeking details about the program through email, Representative shared information with a lead like a brochure of the program, etc.
#     * **Phone Activity:** Had a phone conversation with a representative, had a conversation over SMS with a representative, etc.
#     * **Website Activity:** Interacted on live chat with a representative, updated profile on the website, etc.
# 
# * **print_media_type1:** Flag indicating whether the lead had seen the ad of ExtraaLearn in the Newspaper
# * **print_media_type2:** Flag indicating whether the lead had seen the ad of ExtraaLearn in the Magazine
# * **digital_media:** Flag indicating whether the lead had seen the ad of ExtraaLearn on the digital platforms
# * **educational_channels:** Flag indicating whether the lead had heard about ExtraaLearn in the education channels like online forums, discussion threads, educational websites, etc.
# * **referral:** Flag indicating whether the lead had heard about ExtraaLearn through reference.
# * **status:** Flag indicating whether the lead was converted to a paid customer or not. The class 1 represents the paid customer and class 0 represents the unpaid customer.

# ## **Importing the necessary libraries and overview of the dataset**

# In[1]:


import warnings
warnings.filterwarnings("ignore")

# Libraries for data manipulation and visualization
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

# Algorithms to use
from sklearn.tree import DecisionTreeClassifier

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

# Metrics to evaluate the model
from sklearn.metrics import confusion_matrix, classification_report, recall_score

from sklearn import metrics

# For hyperparameter tuning
from sklearn.model_selection import GridSearchCV


# ### **Loading the dataset**

# In[2]:


learn = pd.read_csv("C:/Users/Anne/OneDrive/Desktop/MIT/Elective Project/ExtraaLearn.csv")


# In[3]:


# Copying data to another variable to avoid any changes to the original data
data = learn.copy()


# ### **View the first and the last 5 rows of the dataset**

# In[4]:


data.head()


# In[5]:


data.tail()


# ### **Understand the shape of the dataset**

# In[6]:


data.shape


# * The dataset has **4612 rows and 15 columns.** 

# ### **Check the data types of the columns in the dataset**

# In[7]:


data.info()


# **Observations:**
# 
# * `age`, `website_visits`, `time_spent_on_website`, `page_views_per_visit`, and `status` are of numeric type while rest of the columns are of object type.
# 
# * There are **no null values** in the dataset.

# In[8]:


# Checking for duplicate values
data.duplicated().sum()


# - There are **no duplicate values** in the data.

# ## **Exploratory Data Analysis**

# ### **Univariate Analysis**

# ### **Question 1:** Write the code to find the summary statistics and write your observations based on that. (4 Marks)

# In[9]:


# Finding Summary statistics
data.describe()


# **Observations:_**
# The summary statistics observe the columns which are numerical 
# 
# 1. age: The average/arithmetic mean for leads is about 46 years old where the median is 51 years. the total range of the group is large, from 18-63 years. 
# 
# 2. website_vsits: The average lead visits the website 3 times. There is a substantial difference between the max visits (30) and the 75th percentile marker (5) which indicates an outlier. This is reaffirmed considering the standard deviation is only about 3 visits, so the max (30) visits is well beyond 3 standard deviations from the mean. 
# 
# 3. time_spent_on_website: The average time spent on the website is 724 seconds, 50% of leads spend 149-1337 seconds on the site. The mean (724) is signifiantly larger than the median (376) which may point out an outlier in the max (2537) which is more than 3 standard deviations from the mean. 
# 
# 4. page_views_per_visit: The average page views per visit is 3, the median is also about 3. 50% of people visit 2-4(3.7) pages in one visit. Also, the max (18) is more than 3 standard deviations from the mean (3), suggesting it is an extreme outlier. 
# 
# 5. status: As this is a binary variable, the mean does not tell us much about the data. We can see however, that 75% of leads are under 1, which means only about 25% of leads convert to paid customers.

# In[10]:


# Making a list of all categorical variables
cat_col = list(data.select_dtypes("object").columns)

# Printing count of each unique value in each categorical column
for column in cat_col:
    print(data[column].value_counts(normalize = True))
    print("-" * 50)


# **Observations:**
# * Most of the leads are working professionals.
# * As expected, the majority of the leads interacted with ExtraaLearn from the website.
# * Almost an equal percentage of profile completions are categorized as high and medium that is 49.1% and 48.6%, respectively. Only **2.3%** of the profile completions are categorized as low.
# * Approx 49.4% of the leads had their last activity over email, followed by 26.8% having phone activity. This implies that the majority of the leads prefer to communicate via email.
# * We can observe that each ID has an equal percentage of values. Let's check the number of unique values in the ID column.

# In[11]:


# Checking the number of unique values
data["ID"].nunique()


# * All the values in the ID column are unique.
# * We can drop this column as it would not add value to our analysis.

# In[12]:


# Dropping ID column
data.drop(["ID"], axis = 1, inplace = True)


# In[13]:


# making a countplot
plt.figure(figsize = (10, 6))

ax = sns.countplot(x = 'status', data = data)

# Annotating the exact count on the top of the bar for each category 
for p in ax.patches:
    ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x(), p.get_height()+ 0.35))


# - The above plot shows that number of leads converted are significantly less than number of leads not converted which can be expected.
# - The plot indicates that **~30%** (1377/4612) of leads have been converted.

# **Let's check the distribution and outliers for numerical columns in the data**

# ### **Question 2: Provide observations for below distribution plots and box plots. (4 Marks)**

# In[14]:


# make a histogram and boxplot for 4 vairables: age, website_visits, time_spent_on_website, and page_views_per_visit
for col in ['age', 'website_visits', 'time_spent_on_website', 'page_views_per_visit']:
    print(col)
    
    print('Skew :',round(data[col].skew(), 2)) # displays the skew under the variable
    
    plt.figure(figsize = (15, 4))
    
    plt.subplot(1, 2, 1)
    
    data[col].hist(bins = 10, grid = False) # histogram
    
    plt.ylabel('count')
    
    plt.subplot(1, 2, 2)
    
    sns.boxplot(x = data[col]) # boxplot
    
    plt.show()


# **Observations:**
# 1. age: The data is slightly negatively skewed which is shown in the histogram by the long tail to the left of the peak of the bell curve. This is also indicated by in the boxplot, the median in the boxplot is closer to the top of the box, and the whisker on the top is shorter than the whisker on the bottom. Where most of the leads fall between 36-57, younger ages above 18 (possibly students or young professionals) also show interest in the program.
# 
# 2. website_visits: The data is extremely positively skewed. The hisogram is truncated and shows a sharp cut off an peak at 0 visits as people cannot go under zero. Also the first bin holds the majority of the leads showing that most leads only visit the site 0-5 times. The skew shows that most of the ouliers fall on the right tail which is demonstrated in the boxplot which shows many outliers in the positive direction of the median.
# 
# 3. time_spent_on_website: The data is positively skewed. The histogram and boxplot show that the middle 50% of users spend 200-1400 seconds on the website.
# 
# 4. page_views_per_visit: The data is positively skewed. The histogram shows most of the data follows a normal distribution but significant outliers in the positive direction are visible on the boxplot. Ignoring the outliers, most leads visit 3 pages while on the website. 

# ### **Bivariate Analysis**

# **We are done with univariate analysis and data preprocessing. Let's explore the data a bit more with bivariate analysis.**
# 
# Leads will have different expectations from the outcome of the course and their current occupation may play a key role for them to take the program. Let's analyze it.

# In[15]:


# plot a coutplot of current_occupation and status
plt.figure(figsize = (10, 6))

sns.countplot(x = 'current_occupation', hue = 'status', data = data)

plt.show()


# **Observations:**
# 
# - The plot shows that working professional leads are more likely to opt for a course offered by the organization and the students are least likely to be converted. 
# - This shows that the currently offered programs are more oriented toward working professionals or unemployed personnel. The programs might be suitable for the working professionals who might want to transition to a new role or take up more responsibility in their current role. And also focused on skills that are in high demand making it more suitable for working professionals or currently unemployed leads.

# **Age can also be a good factor to differentiate between such leads. Let's explore this.**

# In[16]:


# make three boxplots using current occupation and age
plt.figure(figsize = (10, 5))

sns.boxplot(data["current_occupation"], data["age"])

plt.show()


# In[17]:


# find summary statistics for curent_ocupation based on age
data.groupby(["current_occupation"])["age"].describe()


# **Observations:**
# 
# * The range of age for students is 18 to 25 years.
# * The range of age for professionals is 25 to 60 years.
# * The range of age for unemployed leads is 32 to 63 years.
# * The average age of working professionals and unemployed leads is almost 50 years.

# **The company's first interaction with leads should be compelling and persuasive. Let's see if the channels of the first interaction have an impact on the conversion of leads.**

# In[18]:


# plot a countplot of first_interaction and status
plt.figure(figsize = (10, 6))

sns.countplot(x = 'first_interaction', hue = 'status', data = data)

plt.show()


# **Observations:**
# 
# * The website seems to be doing a good job as compared to mobile app as there is a huge difference in the number of conversions of the leads who first interacted with the company through website and those who interacted through mobile application.
# * Majority of the leads who interacted through websites were converted to paid customers, while only a small number of leads, who interacted through mobile app, converted.

# **We observed earlier that some leads spend more time on websites than others. Let's analyze if spending more time on websites results in conversion.**

# ### **Question 3:** 
# - **Create a boxplot for variables 'status' and 'time_spent_on_website' (use sns.boxplot() function) (1 Mark)**
# - **Provide your observations from the plot (2 Marks)**

# In[19]:


# plot a boxplot of status and time_spent_on_website
plt.figure(figsize = (10, 5))

sns.boxplot(data['status'], data['time_spent_on_website']) # boxplots of status 0 and 1 by time 

plt.show()


# **Observations:**
# 1. Those leads who spend between 500-1800 seconds on the website tend to convert to paying customers.
# 2. Leads who spend a very short amount of time 0-600 seconds will likely not convert to paying customers. 
# 3. A long time spent looking through the website may inidcate strong interest in the program or subject which helps to explain higher conversion rates.(It would be valueable to also analyze page_views_per_visit to test this)
# 4. Efforts should be made to keep the lead on the website like a pop up in the case that they try to navigate away, or providing enagging website features. 

# **People browsing the website or the mobile app are generally required to create a profile by sharing their details before they can access more information. Let's see if the profile completion level has an impact on lead coversion**

# In[20]:


plt.figure(figsize = (10, 6))

sns.countplot(x = 'profile_completed', hue = 'status', data = data)

plt.show()


# **Observations:**
# 
# * The leads whose profile completion level is high converted more in comparison to other levels of profile completion.
# * The medium and low levels of profile completion saw comparatively very less conversions.
# * The high level of profile completion might indicate a lead's intent to pursue the course which results in high conversion.

# **Referrals from a converted lead can be a good source of income with a very low cost of advertisement. Let's see how referrals impact lead conversion status.**

# In[21]:


plt.figure(figsize = (10, 6))

sns.countplot(x = 'referral', hue = 'status', data = data)

plt.show()


# **Observations:**
# * There are a very less number of referrals but the conversion is high. 
# * Company should try to get more leads through referrals by promoting rewards for existing customer base when they refer someone.

# ### **Question 4:** Write the code to plot the correlation heatmap and write your observations based on that. (4 Marks)

# In[22]:


# ploting a correlation heatmap of data
plt.figure(figsize = (12, 7))

sns.heatmap(data.corr(), annot = True, fmt = '.2f')
                  
plt.show()


# **Observations:**
# 1. There is a weak positive correlation (.12) between "age" and "status"
# 2. There is a slight positive correlation (.30) between "time_spent_on_website" and "status". This indicates that leads who spend a longer time on the website are more likely to become paying customers.
# 

# ## **Data preparation for modeling**
# 
# - We want to predict which lead is more likely to be converted.
# - Before we proceed to build a model, we'll have to encode categorical features.
# - We'll split the data into train and test sets to be able to evaluate the model that we build on the train data.

# In[23]:


# Separating the target variable and other variables
X = data.drop(columns = 'status')

Y = data['status']


# In[24]:


# Creating dummy variables, drop_first=True is used to avoid redundant variables
X = pd.get_dummies(X, drop_first = True)


# In[25]:


# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 1) #30% used for test


# **Checking the shape of the train and test data**

# In[26]:


print("Shape of the training set: ", X_train.shape)   

print("Shape of the test set: ", X_test.shape)

print("Percentage of classes in the training set:")

print(y_train.value_counts(normalize = True))

print("Percentage of classes in the test set:")

print(y_test.value_counts(normalize = True))


# ## **Building Classification Models**

# **Before training the model, let's choose the appropriate model evaluation criterion as per the problem at hand.**
# 
# ### **Model evaluation criterion**
# 
# **Model can make wrong predictions as:**
# 
# 1. Predicting a lead will not be converted to a paid customer but, in reality, the lead would have converted to a paid customer.
# 2. Predicting a lead will be converted to a paid customer but, in reality, the lead would have not converted to a paid customer. 
# 
# ### **Which case is more important?** 
# 
# * If we predict that a lead will not get converted and the lead would have converted then the company will lose a potential customer. 
# 
# * If we predict that a lead will get converted and the lead doesn't get converted the company might lose resources by nurturing false-positive cases.
# 
# Losing a potential customer is a greater loss for the organization.
# 
# ### **How to reduce the losses?**
# 
# * Company would want `Recall` to be maximized. The greater the Recall score, higher the chances of minimizing False Negatives. 

# **Also, let's create a function to calculate and print the classification report and confusion matrix so that we don't have to rewrite the same code repeatedly for each model.**

# In[27]:


# Function to print the classification report and get confusion matrix in a proper format

def metrics_score(actual, predicted):
    print(classification_report(actual, predicted))
    
    cm = confusion_matrix(actual, predicted)
    
    plt.figure(figsize = (8, 5))
    
    sns.heatmap(cm, annot = True,  fmt = '.2f', xticklabels = ['Not Converted', 'Converted'], yticklabels = ['Not Converted', 'Converted'])
    
    plt.ylabel('Actual')
    
    plt.xlabel('Predicted')
    
    plt.show()


# ### **Decision Tree**

# ### **Question 5:**
# 
# - **Fit the decision tree classifier on the training data (use random_state=7) (2 Marks)**
# - **Check the performance on both training and testing datasets (use metrics_score function) (2 Marks)**
# - **Write your observations (3 Marks)**

# In[28]:


# Fitting the decision tree classifier on the training data
d_tree =  DecisionTreeClassifier(random_state = 7)

d_tree.fit(X_train, y_train)


# **Let's check the performance on the training data**

# In[29]:


# Checking performance on the training data
y_pred_train1 = d_tree.predict(X_train)

metrics_score(y_train, y_pred_train1)


# **Observations:**
# 1. The total size of the training data is 3228 (70% of the entire data set) and the performance is 100% accurate with no false negatives or false positives.
# 2. This implies that the training data is overfitting and may not perform as well on the test or further appliations.

# 
# **Let's check the performance on test data to see if the model is overfitting.**

# In[30]:


# Checking performance on the testing data
y_pred_test1 = d_tree.predict(X_test)

metrics_score(y_test, y_pred_test1)


# **Observations:**
# 1. The decision tree model is overfitting the data and as such did not perform well on the test data.
# 2. The recall score, and f1 score for class 1 is quite low at 70% showing there are many false negatives that were not caught. 

# **Let's try hyperparameter tuning using GridSearchCV to find the optimal max_depth** to reduce overfitting of the model. We can tune some other hyperparameters as well.

# ### **Decision Tree - Hyperparameter Tuning**
# 
# We will use the class_weight hyperparameter with the value equal to {0: 0.3, 1: 0.7} which is approximately the opposite of the imbalance in the original data. 
# 
# **This would tell the model that 1 is the important class here.**

# In[31]:


# Choose the type of classifier 
d_tree_tuned = DecisionTreeClassifier(random_state = 7, class_weight = {0: 0.3, 1: 0.7})

# Grid of parameters to choose from
parameters = {'max_depth': np.arange(2, 10), 
              'criterion': ['gini', 'entropy'],
              'min_samples_leaf': [5, 10, 20, 25]
             }

# Type of scoring used to compare parameter combinations - recall score for class 1
scorer = metrics.make_scorer(recall_score, pos_label = 1)

# Run the grid search
grid_obj = GridSearchCV(d_tree_tuned, parameters, scoring = scorer, cv = 5)

grid_obj = grid_obj.fit(X_train, y_train)

# Set the classifier to the best combination of parameters
d_tree_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data
d_tree_tuned.fit(X_train, y_train)


# We have tuned the model and fit the tuned model on the training data. Now, **let's check the model performance on the training and testing data.**

# ### **Question 6:**
# - **Check the performance on both training and testing datasets (4 Marks)**
# - **Compare the results with the results from the decision tree model with default parameters and write your observations (4 Marks)**

# In[32]:


# Checking performance on the training data
y_pred_train2 = d_tree_tuned.predict(X_train)

metrics_score(y_train, y_pred_train2)


# **Observations:**
# 1. As compared to the performance of the training set with default parameters (which had perfect precision and recall), the performance of the tuned training set has decreased. This is ok as it shows that the tuned training set is no longer overfitting, and the model is able to identify which leads were converted and it does so much better than the basic model operated on the test data as the recall and f1 score is higher.

# **Let's check the model performance on the testing data**

# In[33]:


# Checking performance on the testing data
y_pred_test2 = d_tree_tuned.predict(X_test)

metrics_score(y_test, y_pred_test2)


# **Observations:**
# 1. As compared to the test data set with default parameters, the f1 score for class 1 has increased significantly and is a  stronger model than the un-tuned version.  
# 2. The f1 score has also increased slightly from the basic parameter models. 
# 3. All considered, this model is no longer overfitting and the performance of the tuned training and test sets are much more consistent with eachother than before. It is also showing a promising 86% recall for class 1 which is minimizing the amount of false negatives. This may be a useful model for Extraalearn's needs.

# **Let's visualize the tuned decision tree** and observe the decision rules:

# ### **Question 7: Write your observations from the below visualization of the tuned decision tree. (5 Marks)**

# In[34]:


features = list(X.columns)

plt.figure(figsize = (20, 20))

tree.plot_tree(d_tree_tuned, feature_names = features, filled = True, fontsize = 9, node_ids = True, class_names = True)

plt.show()


# **Note:** Blue leaves represent the converted leads, i.e., **y[1]**, while the orange leaves represent the not converted leads, i.e., **y[0]**. Also, the more the number of observations in a leaf, the darker its color gets.
# 
# **Observations:**
# 1. The first split in the decision tree is "first_interaction_Website" which implies that it is one of the most deciding factors in whether a lead will be converted or not.
# 2. Leads whose first interaction with the company was by website are more likely to convert.  
# 3. Of those leads whose first interaction was by website, the leads who spent over 415.5 seconds on the website were more likely to convert to paying customers.
# 5. And of the leads whose first interaction was by website and who spent over 415.5 seconds on the website, those who were over 25 years old were the most likely in the set to convert to paying customers. 

# **Let's look at the feature importance** of the tuned decision tree model

# In[35]:


# Importance of features in the tree building

print (pd.DataFrame(d_tree_tuned.feature_importances_, columns = ["Imp"], index = X_train.columns).sort_values(by = 'Imp', ascending = False))


# In[36]:


# Plotting the feature importance
importances = d_tree_tuned.feature_importances_

indices = np.argsort(importances)

plt.figure(figsize = (10, 10))

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color = 'violet', align = 'center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()


# **Observations:**
# - **Time spent on the website and first_interaction_website are the most important features** **followed by profile_completed, age, and last_activity**.
# - **The rest of the variables have no impact in this model, while deciding whether a lead will be converted or not**.
# 
# Now, let's build another model - **a random forest classifier.**

# ### **Random Forest Classifier**

# ### **Question 8:** 
# - **Fit the random forest classifier on the training data (use random_state = 7) (2 Marks)**
# - **Check the performance on both training and testing data (use metrics_score function) (2 Marks)**
# - **Write your observations (3 Marks)**

# In[37]:


# Fitting the random forest tree classifier on the training data
rf_estimator = RandomForestClassifier(random_state = 7, criterion = "entropy")

rf_estimator.fit(X_train,y_train)


# **Let's check the performance of the model on the training data**

# In[38]:


# Checking performance on the training data
y_pred_train3 = rf_estimator.predict(X_train)

metrics_score(y_train, y_pred_train3)


# **Observations:**
# 1. The random forest is showing perfect performance on the training set with a f1 score of 1, just like the random forest's training set. This implies it is overfitting once again and performance on the test and further sets could be low. 

# **Let's check the performance on the testing data**

# In[39]:


# Checking performance on the testing data
y_pred_test3 = rf_estimator.predict(X_test)

metrics_score(y_test, y_pred_test3)


# **Observations:**
# 1. The performance is not very good for this model. It is not the best for the company's needs as the recall and macro avg are quite low for class 1 which would increase false negatives and lose the company potential customers. 

# **Let's see if we can get a better model by tuning the random forest classifier**

# ### **Random Forest Classifier - Hyperparameter Tuning**

# Let's try **tuning some of the important hyperparameters of the Random Forest Classifier**. 
# 
# We will **not** tune the `criterion` hyperparameter as we know from hyperparameter tuning for decision trees that `entropy` is a better splitting criterion for this data.

# In[40]:


# Choose the type of classifier
rf_estimator_tuned = RandomForestClassifier(criterion = "entropy", random_state = 7)

# Grid of parameters to choose from
parameters = {"n_estimators": [100, 110, 120],
    "max_depth": [5, 6, 7],
    "max_features": [0.8, 0.9, 1]
             }

# Type of scoring used to compare parameter combinations - recall score for class 1
scorer = metrics.make_scorer(recall_score, pos_label = 1)

# Run the grid search
grid_obj = GridSearchCV(rf_estimator_tuned, parameters, scoring = scorer, cv = 5)

grid_obj = grid_obj.fit(X_train, y_train)

# Set the classifier to the best combination of parameters
rf_estimator_tuned = grid_obj.best_estimator_


# In[41]:


# Fitting the best algorithm to the training data
rf_estimator_tuned.fit(X_train, y_train)


# In[42]:


# Checking performance on the training data
y_pred_train4 = rf_estimator_tuned.predict(X_train)

metrics_score(y_train, y_pred_train4)


# **Observations:**
# - We can see that after hyperparameter tuning, the model is performing poorly on the train data as well.
# - We can try adding some other hyperparameters and/or changing values of some hyperparameters to tune the model and see if we can get better performance.
# 
# **Note:** **GridSearchCV can take a long time to run** depending on the number of hyperparameters and the number of values tried for each hyperparameter. **Therefore, we have reduced the number of values passed to each hyperparameter.** 

# ### **Question 9:**
# - **Tune the random forest classifier using GridSearchCV (4 Marks)**
# - **Check the performance on both training and testing datasets (2 Marks)**
# - **Compare the results with the results from the random forest model with default parameters and write your observations (2 Marks)**

# **Note:** The below code might take some time to run depending on your system's configuration.

# In[43]:


# Choose the type of classifier 
rf_estimator_tuned = RandomForestClassifier(criterion = "entropy", random_state = 7)

# Grid of parameters to choose from
parameters = {"n_estimators": [110, 120],
    "max_depth": [6, 7],
    "min_samples_leaf": [20, 25],
    "max_features": [0.8, 0.9],
    "max_samples": [0.9, 1],
    "class_weight": ["balanced",{0: 0.3, 1: 0.7}]
             }

# Type of scoring used to compare parameter combinations - recall score for class 1
scorer = metrics.make_scorer(recall_score, pos_label = 1)

# Run the grid search on the training data using scorer=scorer and cv=5
grid_obj = GridSearchCV(rf_estimator_tuned, parameters, scoring = scorer, cv = 5)

grid_obj = grid_obj.fit(X_train, y_train)

# Save the best estimator to variable rf_estimator_tuned
rf_estimator_tuned = grid_obj.best_estimator_

#Fit the best estimator to the training data
rf_estimator_tuned.fit(X_train, y_train)


# **Let's check the performance of the tuned model**

# In[44]:


# Checking performance on the training data
y_pred_train5 = rf_estimator_tuned.predict(X_train)

metrics_score(y_train, y_pred_train5)


# **Observations:**
# 1. The training set for the RFM is very simiar to the hyper-tuned set, sharing the same  f1 scores.
# 2. This model shows a promising recall score for class one (87%) and a strong macro avg of 85%/
# 3. Compared to the trainings set on the random forest model with default parameters, this model has a lower f1 score as the default parameter model was over fitting.

# **Let's check the model performance on the test data**

# In[45]:


# Checking performance on the test data
y_pred_test5 = rf_estimator_tuned.predict(X_test)

metrics_score(y_test, y_pred_test5)


# **Observations:**
# 1. The test and training random forest models that have been tuned have very similar performance. 
# 2. Compared to the RFM with default parameters, the tuned RFM shows performance that is less likely to overfit the data and could be a potential model to identify which leads will convert.  
# 3. This model would also match the company's goals of reducing false negatives, as the recall for class 1 is 85-87% and the test macro avg is 84%.
# 

# **One of the drawbacks of ensemble models is that we lose the ability to obtain an interpretation of the model. We cannot observe the decision rules for random forests the way we did for decision trees. So, let's just check the feature importance of the model.**

# In[46]:


importances = rf_estimator_tuned.feature_importances_

indices = np.argsort(importances)

feature_names = list(X.columns)

plt.figure(figsize = (12, 12))

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color = 'violet', align = 'center')

plt.yticks(range(len(indices)), [feature_names[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()


# 
# **Observations:**
# - Similar to the decision tree model, **time spent on website, first_interaction_website, profile_completed, and age are the top four features** that help distinguish between not converted and converted leads.
# - Unlike the decision tree, **the random forest gives some importance to other variables like occupation, page_views_per_visit, as well.** This implies that the random forest is giving importance to more factors in comparison to the decision tree.

# ## **Conclusion and Recommendations**

# ### **Question 10:**
# 
# **Write your conclusions on the key factors that drive the conversion of leads and write your recommendations to the business on how can they improve the conversion rate. (10 Marks)**

# ### **Conclusions:** conclusions on the key factors that drive the conversion of leads 
# 1. The tree-based models developed in this case study can be used to predict which leads will be converted into paying customers and they can also be used to determine what steps should be taken to  raise this coversion rate.
# 
# 2. The decision tree model with basic parameters is highly overfitting the training data and as such gives a low Recall and F1 score 70% on the test data. 
# 
# 3. The tuned decision tree model is very balanced and gives more generalized results on both training and testing data. It also has a high recall (86-88%) for class 1 which alligns with the company's goal to minimize the leads that are falsely predicted to not enroll. This model can be used to predict which leads will convert.
# 
# 4. The RFM with basic parameters is giving the highest F1 score of 90% and a macro average of 83% on the test data. This model is overfitting and also performs poorly on the training data, returning a low recall score for class 1.
# 
# 5. The tuned RFM could also be used to predict which leads will convert as it is close to balanced and returned a high recall (85-87%) for class 1 in the test and practice data. It also output a gave a macro average of 82% on the test and training data.
# 
# 6. Based on the feature importances graph, "time_spent_on_website", "first_interaction_website", "profile_completed", and perhaps "age" , are the four driving variables of the model.
# 

# ### **Business Recommendations** recommendations to the business on how can they improve the conversion rate
# 1. Considering that "time_spent_on_website" is a leading factor in determining whether a lead convert to a paying student,  and that specifically those leads who spend over 7 minutes are the most likely to convert, it is in ExtraaLearn's best interest to increase the amount of time leads spend on the website. This can be done using numerous approaches.
#    
#    a. Interesting content. Increase the amount of enagging content on the website to grab the leads attention. Ways to do this include: specific timelines of the program, example projects from previous years, testimonials from various backgrounds, and student reviews.
#    
#    b. interactive content. Creating interactive content allows leads to get familiar with the program. Ways to do this include: a pre-test/sample problems, interesting short clip from a lecture, chat bot to answer questions.
#     
#     c. Popup when lead navigates away from webpage. A lead leaving the page too soon can be prevented by a popup that appears when they try to navigate away from the site. This popup could be a countdown to the start of the next program, an option to put in contact info for more information regarding the program, or a request to create a profile to save the lead's progress. This makes the lead reconsider leaving the site soon and may convert a missed potential lead to a paying student.
#     
#     
# 2. The starting leaf on the vizualization of the decision tree was "first_interaction_website" which means it is valuable to direct leads to the website before other sources of information like an app,a s most of those leads who use the website as a primary source of information convert to students. This can be done by running ads for the program on professional networking websites such as LinkedIn or Facebook, where it would be easy for the lead to click on a link directing them to Extraalearn's website. 
# 
# 
# 3. Another important variable is "profile_completed_medium". Leads with profiles above 50% completion are likely to convert. Any steps made toward encouraging leads to complete profiles would be beneficial. Having an inscentive for completing your profile like pre-access to one short lecture could raise the number of completed profiles. 
# 
# 
# 4. Where the "age" value does not have as much influence on the status of leads as the previous three variables, it is still worth exploring as it is extremely costly to lose a potential customer, and the bivariate analysis showed that working professionals were more likely to opt for the course than students or unemployed adults. And from the decision tree visualization we learned that one of the demographics that is most likely to convert is adults over 25 years old who first interacted with ExtraaLearn through the website and who spent more than 7 minutes on the website. To find people in this demographic, Extraalearn could partner with companies to have access to more professional leads while offering their service to the company at a bulk cost for group training. This increases access to the target demographic for Extraalearn and the company would be able to upskill/reskill their workforce.
# 
# 
# 5. Referals were shown to be an effective way of advertizing the course while boosting conversion rates. Offer cash benefits (return a portion of the course tuition) for students who refer a lead who then converts to a paying customer.  
