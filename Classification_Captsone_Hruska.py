#!/usr/bin/env python
# coding: utf-8

# # **Milestone 1**

# ## <b>Problem Definition</b>
# 
# **The context:**
# 1. A large portion of bank profits come from interests on home loans. Defaulters cause banks to lose a large portion of their profits. 
# 2. The manual process of approving loans is prone to human error and biases.
# 3. Creating a method of loan approval which utilizes  machine learning could remove human biases from the process.
# 
# **The objectives:**
# 1. Create a model to automate the approval process  for loans.
# 2. The model should accurately predict clients who are likely to default on their loan.
# 3. The model should give recommendations to the bank on important features to the approval process.
# 
# **The key questions:** 
# 1. Which individual variables have the greatest impact on whether a loan will default?
# 2. What clients are the most likely  to default a loan?
# 3. What biases were introduced/trends appeared due to the human approval process? 
# <br>
# **The problem formulation:**
# 
# How can data science and machine learning be utilized to make a model that will  accurately predict loan defaulting of clients? This model should be free of any biases that have been introduced due to the manual approval process and should provide sufficient explanation for any rejected cases.
# 
# 
# ## **Data Description:**
# The Home Equity dataset (HMEQ) contains baseline and loan performance information for 5,960 recent home equity loans. The target (BAD) is a binary variable that indicates whether an applicant has ultimately defaulted or has been severely delinquent. This adverse outcome occurred in 1,189 cases (20 percent). 12 input variables were registered for each applicant.
# 
# 
# * **BAD:** 1 = Client defaulted on loan, 0 = loan repaid
# 
# * **LOAN:** Amount of loan approved.
# 
# * **MORTDUE:** Amount due on the existing mortgage.
# 
# * **VALUE:** Current value of the property. 
# 
# * **REASON:** Reason for the loan request. (HomeImp = home improvement, DebtCon= debt consolidation which means taking out a new loan to pay off other liabilities and consumer debts) 
# 
# * **JOB:** The type of job that loan applicant has such as manager, self, etc.
# 
# * **YOJ:** Years at present job.
# 
# * **DEROG:** Number of major derogatory reports (which indicates a serious delinquency or late payments). 
# 
# * **DELINQ:** Number of delinquent credit lines (a line of credit becomes delinquent when a borrower does not make the minimum required payments 30 to 60 days past the day on which the payments were due). 
# 
# * **CLAGE:** Age of the oldest credit line in months. 
# 
# * **NINQ:** Number of recent credit inquiries. 
# 
# * **CLNO:** Number of existing credit lines.
# 
# * **DEBTINC:** Debt-to-income ratio (all your monthly debt payments divided by your gross monthly income. This number is one way lenders measure your ability to manage the monthly payments to repay the money you plan to borrow.

# ## <b>Important Notes</b>
# 
# - This notebook can be considered a guide to refer to while solving the problem. The evaluation will be as per the Rubric shared for each Milestone. Unlike previous courses, it does not follow the pattern of the graded questions in different sections. This notebook would give you a direction on what steps need to be taken in order to get a viable solution to the problem. Please note that this is just one way of doing this. There can be other 'creative' ways to solve the problem and we urge you to feel free and explore them as an 'optional' exercise. 
# 
# - In the notebook, there are markdowns cells called - Observations and Insights. It is a good practice to provide observations and extract insights from the outputs.
# 
# - The naming convention for different variables can vary. Please consider the code provided in this notebook as a sample code.
# 
# - All the outputs in the notebook are just for reference and can be different if you follow a different approach.
# 
# - There are sections called **Think About It** in the notebook that will help you get a better understanding of the reasoning behind a particular technique/step. Interested learners can take alternative approaches if they want to explore different techniques. 

# ### **Import the necessary libraries**

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score,precision_score,recall_score,f1_score

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

import scipy.stats as stats

from sklearn.model_selection import GridSearchCV


import warnings
warnings.filterwarnings('ignore')


# ### **Read the dataset**

# In[2]:


# Reading the dataset from file
hm=pd.read_csv("C:/Users/Anne/OneDrive/Desktop/MIT/Capstone/hmeq.csv")


# In[3]:


# Copying data to another variable to avoid any changes to original data
data = hm.copy()


# ### **Print the first and last 5 rows of the dataset**

# In[4]:


# Display first five rows

data.head()


# In[5]:


# Display last 5 rows
data.tail()


# ### **Understand the shape of the dataset**

# In[6]:


# Check the shape of the data
data.shape


# **Insights:**
# 1. The data set has 5960 rows and 13 columns

# ### **Check the data types of the columns**

# In[7]:


# Check info of the data
data.info()


# **Insights:**
# 1. "Reason" and "Job" are the only two columns which are of object type, the remaining 11 columns are numeric. "BAD" and "LOAN" are the only two numeric types that are integers, the rest are floats.
# 2. There are missing values in every column besides "BAD" and "LOAN" which return 5960 non-null values.

# ### **Check for missing values**

# In[8]:


# Analyse missing values - Hint: use isnull() function
data.isnull().sum()


# In[9]:


# Check the percentage of missing values in the each column.
# Hint: divide the result from the previous code by the number of rows in the dataset
percent_missing = data.isnull().sum() * 100 / len(data)
print(percent_missing)


# **Insights:**
# 1. Columns "BAD" and "LOAN"  have no missing values.
# 2. "VALUE" has only 1% of its data missing, where columns "DEBTINC"  and "DEROG" have the most missing data with 21% and ~12% mising respectively. Though they have a high amount of missing data, the variables are important and should be kept.  

# ### **Think about it:**
# - We found the total number of missing values and the percentage of missing values, which is better to consider? The Percentage of missing values gives us a better idea of the proportional loss of data. If that loss (percentage) is too high, it may be necessary to get rid of that feature.
# - What can be the limit for % missing values in a column in order to avoid it and what are the challenges associated with filling them and avoiding them? 
# 20% of data is reasonable to miss, unless the variable is important.

# **We can convert the object type columns to categories**
# 
# `converting "objects" to "category" reduces the data space required to store the dataframe`

# ### **Convert the data types**

# In[10]:


cols = data.select_dtypes(['object']).columns.tolist()

# adding target variable to this list as this is an classification problem and the target variable is categorical
# adding 'BAD' to the list
cols.append('BAD')


# In[11]:


cols


# In[12]:


# Changing the data type of object type column to category. hint use astype() function
for i in cols:
    data[i] = data[i].astype('category')


# In[13]:


# Checking the info again and the datatype of different variable
data.info()


# ### **Analyze Summary Statistics of the dataset**

# In[14]:


# Analyze the summary statistics for numerical variables
data.describe()


# **Insights**
# 
# 1. LOAN - The avergae/arithmetic mean for loans given out is 18,607 dollars. The range for loans given out is large spanning from 1,100 to 89,900 dollars.
# 
# 2. MORTDUE - The average mortgage due is 73,760 dollars. There is a substantial difference between the mean (73,760) and the max mortgage due (399,550), this difference is more than 3 standard deviations (44,457), which may indicate an outlier.  
# 
# 3. VALUE - The average current value of a property is 57,385 dollars.
# 
# 4. YOJ - The average years a client spends at their current job is 8.9. 
# 
# 5. DEROG - The average number of major derogatory reports is .25, which means most clients do not have major derogatory reports. This is affirmed by the 75% percentile (0) which states that 75% of clients do not have derogatory reports. The max number of derogatory reports that a client has, however, is 10.  
# 
# 6. DELINQ - The average number of delinquent credit lines a client has is  .45, which similar to "DEROG" suggests that most clients have fewer than 1 delinquent credit line. This is confirmed by the 75% percentile (0) which states that 75% of clients do not have a delinquent credit line. The max number of delinquent credit lines however, is 15. 
# 
# 7. CLAGE - The average age of the oldest credit line for a client is about 180 months, or 15 years. The maximum value (about 97 years) could indicate a possible outlier as it is way more than 3 standard deviations from the mean. 
# 
# 8. NINQ - The average number of credit inquiries for a client is about 1. 75% of clients have fewer than 2 credit inquiries on their account. 
# 
# 9. CLNO - The average number of exisiting credit lines for a client is 21.
# 
# 10. DEBTINC - The average debt-to-income ratio for clients is nearly 34%. 75% of clients have a debt-to-income ratio below 39%. The max DTI ratio of 203 could signify an outlier as it is many standard deviations away from the mean and the median dti. 

# In[15]:


# Check summary for categorical data - Hint: inside describe function you can use the argument include=['category']
data.describe(include=['category']).T #.T tranposes the results such that the categorical data becomes the rows


# **Insights**
# 1. BAD - 80% of clients have not defaulted a loan. 
# 2. REASON - the majority of clients (about 69%) have requested a loan to pay off other liabilities and consumer debts.
# 3. JOB - the majority of clients (42%) work in a field not specified, where the others work in one of 5 pre-specified areas.

# **Let's look at the unique values in all the categorical variables**

# In[16]:


# Checking the count of unique values in each categorical column 
cols_cat= data.select_dtypes(['category'])

for i in cols_cat.columns:
    print('Unique values in',i, 'are :')
    print(cols_cat[i].unique())
    print('*'*40)


# **Insights**
# 1. "BAD" has two unique values: 0 and 1 which are both ints
# 2. "REASON" has two unique values: 'HomeImp' and 'DebtCon' which are objects as well as missing values under 'NaN'.
# 3. "JOB" has six unique values: 'Mgr', 'Office', 'Other', 'ProfExe', 'Sales', 'Self' which are objects and also missing values under 'NaN'.

# ### **Think about it**
# - The results above gave the absolute count of unique values in each categorical column. Are absolute values a good measure? 
#     It is good for getting an idea of what the possible outcomes are, but it provides little useful information.
#     
# - If not, what else can be used? Try implementing that. 
#     Displaying the percentages of each unique value in each category.

# In[17]:


cols_cat= data.select_dtypes(['category'])

for i in cols_cat.columns:
    print('Unique values in',i, 'are :')
    print(cols_cat[i].value_counts()/cols_cat[i].count()) #displays the percentage of unique values as a percentages between 0 and 1
    print('*'*40)


# ## **Exploratory Data Analysis (EDA) and Visualization**

# ## **Univariate Analysis**
# 
# Univariate analysis is used to explore each variable in a data set, separately. It looks at the range of values, as well as the central tendency of the values. It can be done for both numerical and categorical variables

# ### **1. Univariate Analysis - Numerical Data**
# Histograms and box plots help to visualize and describe numerical data. We use box plot and histogram to analyze the numerical columns.

# In[18]:


# While doing uni-variate analysis of numerical variables we want to study their central tendency and dispersion.
# Let us write a function that will help us create boxplot and histogram for any input numerical variable.
# This function takes the numerical column as the input and return the boxplots and histograms for the variable.
# Let us see if this help us write faster and cleaner code.
def histogram_boxplot(feature, figsize=(15,10), bins = None):
    """ Boxplot and histogram combined
    feature: 1-d feature array
    figsize: size of fig (default (9,8))
    bins: number of bins (default None / auto)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(nrows = 2, # Number of rows of the subplot grid= 2
                                           sharex = True, # x-axis will be shared among all subplots
                                           gridspec_kw = {"height_ratios": (.25, .75)}, 
                                           figsize = figsize 
                                           ) # creating the 2 subplots
    sns.boxplot(feature, ax=ax_box2, showmeans=True, color='violet') # boxplot will be created and a star will indicate the mean value of the column
    sns.distplot(feature, kde=F, ax=ax_hist2, bins=bins,palette="winter") if bins else sns.distplot(feature, kde=False, ax=ax_hist2) # For histogram
    ax_hist2.axvline(np.mean(feature), color='green', linestyle='--') # Add mean to the histogram
    ax_hist2.axvline(np.median(feature), color='black', linestyle='-') # Add median to the histogram


# #### Using the above function, let's first analyze the Histogram and Boxplot for LOAN

# In[19]:


# Build the histogram boxplot for Loan
histogram_boxplot(data['LOAN'])


# **Insights**
# 1. The boxplot for "LOAN" confirm that there are numerous oultiers in the positive direction, which is a trend noticed from the summary statistics.
# 2. Both the boxplot and the histogram indicate that LOAN data is slightly positively skewed. In the box plot this is shown by the median line closer to the bottom of the box plot, and the whisker on the top is much longer than the whisker on the bottom. In the histogram this is shown by a longer tail to the right of the peak of the bell curve. 

# #### **Note:** As done above, analyze Histogram and Boxplot for other variables

# In[20]:


histogram_boxplot(data['MORTDUE'])


# In[21]:


histogram_boxplot(data['VALUE'])


# In[22]:


histogram_boxplot(data['YOJ'])


# In[23]:


histogram_boxplot(data['CLAGE'])


# In[24]:


histogram_boxplot(data['DEBTINC'])


# **Insights**
# 1. MORTDUE - The histogram displays a good normal distribution though there is a slight positive skew. The boxplot shows many outliers in the positive direction, and that the bottom whisker is closer to the box than the top whisker.
# 2. VALUE - Value is also positively skewed, with most outliers shown past the top whisker in the boxplot. And the histogram has a longer right tail.
# 3. YOJ - Years at present jobs demonstrates a strong postive skew and the histogram is trunkated at 0 as no one can work fewer than 0 years. The boxplot also shows the skew through a long top whisker and many outliers in the positive direction
# 4. CLAGE - Age of the oldest credit line shows distribution close to normal distribution in the histogram, though the boxplot confirms there is a very small positive skew. 
# 5. DEBTINC - Debt-to-income ratio demonstrates a normal distribution in the histogram, with the boxplot also showing a very normal trend with the median nearly perfectly alligned with the confidence interval.  

# ### **2. Univariate Analysis - Categorical Data**

# In[25]:


# Function to create barplots that indicate percentage for each category.

def perc_on_bar(plot, feature):
    '''
    plot
    feature: categorical feature
    the function won't work if a column is passed in hue parameter
    '''

    total = len(feature) # length of the column
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total) # percentage of each class of the category
        x = p.get_x() + p.get_width() / 2 - 0.05 # width of the plot
        y = p.get_y() + p.get_height()           # height of the plot
        ax.annotate(percentage, (x, y), size = 12) # annotate the percentage 
        
    plt.show() # show the plot


# #### Analyze Barplot for DELINQ

# In[26]:


#Build barplot for DELINQ

plt.figure(figsize=(15,5))
ax = sns.countplot(data["DELINQ"],palette='winter')
perc_on_bar(ax,data["DELINQ"])


# **Insights**
# 1. The vast majority (70%) of clients have 0 delinquent credit lines. 
# 2. The number of delinquent credit lines decreases dramatically, with only 2.7% of clients having 4 or more delinquencies

# #### **Note:** As done above, analyze barplots for other variables.

# In[27]:


plt.figure(figsize=(15,5))
ax = sns.countplot(data["BAD"],palette='winter')
perc_on_bar(ax,data["BAD"])


# In[28]:


plt.figure(figsize=(15,5))
ax = sns.countplot(data["REASON"],palette='winter')
perc_on_bar(ax,data["REASON"])


# In[29]:


plt.figure(figsize=(15,5))
ax = sns.countplot(data["JOB"],palette='winter')
perc_on_bar(ax,data["JOB"])


# In[30]:


plt.figure(figsize=(15,5))
ax = sns.countplot(data["DEROG"],palette='winter')
perc_on_bar(ax,data["DEROG"])


# In[31]:


plt.figure(figsize=(15,5))
ax = sns.countplot(data["NINQ"],palette='winter')
perc_on_bar(ax,data["NINQ"])


# **Insights**
# 1. BAD - The vast majority (80%) of clients have not defaulted their loan.
# 2. REASON - Most (66%) of clients take out a loan to pay off other liabilities and consumer debts.
# 3. JOB - Most clients (40%) work in a category not specified. The second largest category is 21% of clients who work as ProfExe
# 4. DEROG - 76% of clients have 0 derogatory reports.
# 5. NINQ - Most clients (42.5%) have 0 recent credit inquiries,22% have 1 inqiry. 

# ## **Bivariate Analysis**

# ###**Bivariate Analysis: Continuous and Categorical Variables**

# #### Analyze BAD vs Loan

# In[32]:


sns.boxplot(data["BAD"],data['LOAN'],palette="PuBu")


# **Insights**
# 1. BAD vs LOAN - Looking at the boxes (IQR)  shows that the there is a difference between variables as non-defaulters have a higher avg loan and those who defaulted have a lower average loan. The negative case (BAD = 0) also displays more outliers with loans over 80,000.

# ####**Note:** As shown above, perform Bi-Variate Analysis on different pair of Categorical and continuous variables

# In[33]:


sns.boxplot(data["BAD"],data['MORTDUE'],palette="PuBu")


# In[34]:


sns.boxplot(data["BAD"],data['VALUE'],palette="PuBu")


# In[35]:


sns.boxplot(data["BAD"],data['YOJ'],palette="PuBu")


# In[36]:


sns.boxplot(data["BAD"],data['CLAGE'],palette="PuBu")


# In[37]:


sns.boxplot(data["BAD"],data['DEBTINC'],palette="PuBu")


# **Insights**
# 1. BAD vs MORTDUE -Those who have not defaulted (BAD = 0) tend to have a higher amount due on their existing mortage, though the two boxplots are very similar. Those who default tend to have more extreme outliers.
# 2. BAD vs VALUE - These boxplots are very similar and do not show many ways to distinguish themselves, though the positive case (BAD = 1) displays more outliers and they are more extreme in the positive direction. 
# 3. BAD vs YOJ - There is a slight difference between the two boxplots, those clients who did not default show an average longer time worked at their present job. 
# 4. BAD vs CLAGE - Those clients who do not default tend to have a higher age for the age of their oldest credit line.
# 5. BAD vs DEBTINC - Those clients who do not default have a lower debt-to-income ratio. Those with a debt-to-income ratio over 50 tend to default.

# ### **Bivariate Analysis: Two Continuous Variables**

# In[38]:


sns.scatterplot(data["VALUE"],data['MORTDUE'],palette="PuBu")


# **Insights:**
# 1. The graph of "Value of the property" vs "Amount due on exisiting mortgage" displays a positive trend and strong positive correlation. 

# #### **Note:** As shown above, perform Bivariate Analysis on different pairs of continuous variables

# In[39]:


sns.scatterplot(data["VALUE"],data['LOAN'],palette="PuBu")


# In[40]:


sns.scatterplot(data["MORTDUE"],data['LOAN'],palette="PuBu")


# **Insights**
# 1. VALUE vs LOAN - This scatterplot shows a slight positive correlation between 'VALUE' and 'LOAN'
# 2. MORTDUE vs LOAN - This scatterplot shows a very slight positive correlation between 'MORTDUE' and 'LOAN' 

# ### **Bivariate Analysis:  BAD vs Categorical Variables**

# **The stacked bar chart (aka stacked bar graph)** extends the standard bar chart from looking at numeric values across one categorical variable to two.

# In[41]:


### Function to plot stacked bar charts for categorical columns

def stacked_plot(x):
    sns.set(palette='nipy_spectral')
    tab1 = pd.crosstab(x,data['BAD'],margins=True)
    print(tab1)
    print('-'*120)
    tab = pd.crosstab(x,data['BAD'],normalize='index')
    tab.plot(kind='bar',stacked=True,figsize=(10,5))
    plt.legend(loc='lower left', frameon=False)
    plt.legend(loc="upper left", bbox_to_anchor=(1,1))
    plt.show()


# #### Plot stacked bar plot for for LOAN and REASON

# In[42]:


# Plot stacked bar plot for BAD and REASON
stacked_plot(data['REASON'])


# **Insights**
# 1. BAD vs REASON- As the stacked barplot shows, reason does not significantly differentite what percent of clients will default. Both DebtCon and HomeImp have just about the same percent of clients within that category who default vs those who do not. 

# #### **Note:** As shown above, perform Bivariate Analysis on different pairs of Categorical vs BAD

# In[43]:


stacked_plot(data['JOB'])


# In[44]:


stacked_plot(data['DEROG'])


# In[45]:


stacked_plot(data['DELINQ'])


# In[46]:


stacked_plot(data['NINQ'])


# In[47]:


stacked_plot(data['CLNO'])


# **Insights**
# 1. BAD vs JOB - Clients who work in sales are most likely to default their loan, second most likely to default are those who are self employed. 
# 2. BAD vs DEROG - Number of major derogatory reports is strongly related to a client's chance of defaulting, with 100% of the clients with 7 or more derogatory reports defaulting.  
# 3. BAD vs DELINQ - Number of delinquent credit lines is stromgly related to a client's chance of defaulting with 100% of clients with 6 delinquent credit lines or more defaulting.
# 4. BAD vs NINQ - Number of recent credit inquiries shows a correltation to clients who default, with 100% of clients with 12 or more recent credit inquries defaulting their loan.
# 5. BAD vs CLNO - Those clients who have many existing credit lines are more likely to default their loans than those who have under 55. 

# ### **Multivariate Analysis**

# #### Analyze Correlation Heatmap for Numerical Variables

# In[48]:


# Separating numerical variables
numerical_col = data.select_dtypes(include=np.number).columns.tolist()

# Build correlation matrix for numerical columns
corr = data[numerical_col].corr()

# plot the heatmap
plt.figure(figsize=(16,12))
sns.heatmap(corr,cmap='coolwarm',vmax=1,vmin=-1,
        fmt=".2f",
        xticklabels=corr.columns,
        yticklabels=corr.columns);


# In[49]:


# Build pairplot for the data with hue = 'BAD'
sns.pairplot(data, hue= 'BAD')


# ### **Think about it**
# - Are there missing values and outliers in the dataset? If yes, how can you treat them?
# VALUE" has only 1% of its data missing, where columns "DEBTINC"  and "DEROG" have the most missing data with 21% and ~12% missing respectively. Though they have a high amount of missing data, the variables are important and should be kept
# 
# - Can you think of different ways in which this can be done and when to treat these outliers or not?
# These outliers should be treated when over 1.5x the IQR when greater than Q3 or less than Q1.
# 
# - Can we create new features based on Missing values?
# Yes, these fatures can be flags to signal whether or not data is missing. 

# #### Treating Outliers

# In[50]:


def treat_outliers(df,col):
    '''
    treats outliers in a varaible
    col: str, name of the numerical varaible
    df: data frame
    col: name of the column
    '''
    
    Q1= np.percentile(df[col] , .25) # 25th quantile
    Q3= np.percentile(df[col], .75)  # 75th quantile
    
    IQR= Q3 - Q1   # IQR Range
    Lower_Whisker = Q1 - (1.5 * IQR)  #define lower whisker
    Upper_Whisker = Q3 + (1.5 * IQR)  # define upper Whisker
    df[col] = np.clip(df[col], Lower_Whisker, Upper_Whisker) # all the values samller than Lower_Whisker will be assigned value of Lower_whisker 
                                                            # and all the values above upper_whishker will be assigned value of upper_Whisker 
    return df

def treat_outliers_all(df, col_list):
    '''
    treat outlier in all numerical varaibles
    col_list: list of numerical varaibles
    df: data frame
    '''
    for c in col_list:
        df = treat_outliers(df,c)
        
    return df
    


# In[51]:


df_raw = data.copy()

numerical_col = df_raw.select_dtypes(include=np.number).columns.tolist()# getting list of numerical columns

df = treat_outliers_all(df_raw,numerical_col)


# #### Adding new columns in the dataset for each column which has missing values 

# In[52]:


#For each column we create a binary flag for the row, if there is missing value in the row, then 1 else 0. 
def add_binary_flag(df,col):
    '''
    df: It is the dataframe
    col: it is column which has missing values
    It returns a dataframe which has binary falg for missing values in column col
    '''
    new_col = str(col)
    new_col += '_missing_values_flag'
    df[new_col] = df[col].isna()
    return df


# In[53]:


# list of columns that has missing values in it
missing_col = [col for col in df.columns if df[col].isnull().any()]

for colmn in missing_col:
    add_binary_flag(df,colmn)
    


# #### Filling missing values in numerical columns with median and mode in categorical variables

# In[54]:


#  Treat Missing values in numerical columns with median and mode in categorical variables
# Select numeric columns.
num_data = df.select_dtypes('number')

# Select string and object columns.
cat_data = df.select_dtypes('category').columns.tolist()#df.select_dtypes('object')

# Fill numeric columns with median.
df[num_data.columns] = num_data.fillna(df.median()) #fill NaN values in all columns with median


# Fill object columns with mode.
for column in cat_data:
    mode = df[column].mode()[0]
    df[column] = df[column].fillna(df[column].mode()[0])


# In[55]:


# check that there are no null values
df.info()


# In[56]:


#Data is the DataFrame name, which you can change according to your notebook
df.to_csv("cleaned_data.csv", index=False)


# ## **Proposed approach**
# **1. Potential techniques** -
# 1. Regression techniques could be applied to predict potential loan defaults. 
# 2. Classification and logistic regression can be used to take the 12 changing variables that the clients have and predict whether their loan will default or not
# 
# **2. Overall solution design** 
# 1. A test and training set of data could be created from the  HMEQ dataset. These can then be used in linear regression with various variables of interest.
#  
# 2. Create a model that can accurately synthesize the 12 variables and process them for each client to give an estimate between 0 and 1 of how likely a specific client is to default
# 
# 
# **3. Measures of success** - What are the key measures of success?
# 1. Statistical accuracy is the most important measure of success. How accurately can the model predict whether a new client will default on a loan. 
# 2. The model should be free from human biases.
# 3. The model performance should be efficient and pass model evaluation metrics like confusion matrices. 

# # **Milestone 2**

# ## **Model Building - Approach**
# 1. Data preparation
# 2. Partition the data into train and test set
# 3. Fit on the train data
# 4. Tune the model and prune the tree, if required
# 5. Test the model on test set

# ## **Data Preparation**

# In[1]:


# importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score,precision_score,recall_score,f1_score,precision_recall_curve

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

import scipy.stats as stats

from sklearn.model_selection import GridSearchCV


import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Reading the dataset from file
milestone1_data = pd.read_csv("cleaned_data.csv")


# In[3]:


# copying milestone1 data on to new variable "data"
data = milestone1_data.copy()


# In[4]:


# check that 'data' has no missing values
data.info()


# ### **Separating the target variable from other variables**

# In[5]:


# Drop the dependent variable from the dataframe and create the X(independent variable) matrix
x = data.drop(columns = 'BAD') # all columns excluding BAD

# Create dummy variables for the categorical variables - Hint: use the get_dummies() function
x = pd.get_dummies(x, drop_first = True)

# Create y(dependent varibale)
y = data['BAD']  # BAD is the dependent variable


# ### **Splitting the data into 70% train and 30% test set**

# In[6]:


# Split the data into training and test set
#STRATIFICATION??

#train_test_split(..., stratify = y)
#y = 0 - 84
#x = 1 - 16
#y_train = 0 - 84%
#x_train = 1 - 16%

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 1)


# ### **Think about it** 
# - You can try different splits like 70:30 or 80:20 as per your choice. Does this change in split affect the performance?
# - If the data is imbalanced, can you make the split more balanced and if yes, how?
# 
# 1. Studies have shown that splitting the data such that 20-30% of the data is used for testing and the rest for training is optimal and minimizes wasted data. Based on teh size of the dataset and how much data we can afford to lose in order to make an better mdoel. In this case delegating 30% of teh data to be used for training is effective and not wasteful. 20% could be used but it may result in a poorer model.
# 2. If a data set is highly unbalanced you can use resampling on the training data or a SMOTE technique to double the sample size of the minority data (in our case defaulters). Our set is unbalanced such that clients who have repayed their loans make up the majority of the data and the data set may benefit from techniques like SMOTE.

# ## **Model Evaluation Criterion**
# 
# #### After understanding the problem statement, think about which evaluation metrics to consider and why. 
# 1. Predict that a client will not default but in reality the client would default. (False Negative)
#     - If we predict that a client will not default but in reality they do, then the bank loses money due to that error, and that client could be a continued risk. 
#     
# 2. Predict that a client will default but in reality the client would not.
#     - If we predict that a client will default but in reality they do not, then the bank loses a potential customer.
#     
# It is best for Recall to be Maximized, as the greater the Recall score the higher chance of minimizing the False Negative scenario.

# In[7]:


#creating metric function 
def metrics_score(actual, predicted):
    print(classification_report(actual, predicted))
    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8,5))
    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels=['Repays', 'Defaults'], yticklabels=['Repays', 'Defaults'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


# ### **Build a Logistic Regression Model** 

# In[8]:


# Defining the Logistic regression model
lg = LogisticRegression()

# Fitting the model on the training data 
lg.fit(x_train, y_train)


# #### Checking the performance on the train dataset

# In[9]:


#Predict for train set
y_pred_train = lg.predict(x_train)


#checking the performance on the train dataset
metrics_score(y_train, y_pred_train)


# #### Checking the performance on the test dataset

# In[10]:


#Predict for test set
y_pred_test = lg.predict(x_test)

#checking the performance on the test dataset
metrics_score(y_test, y_pred_test)


# **Observations:**
# 1. The accuracy for the train and test sets are consistently around 80-85%.
# 2. Recall for class 0 is over fit for both the test and train datasets, as demonstrated by the 100% rate in both.
# 2. Recall for class 1 (the instance of defaulters) is incredibly low, almost 0 for both test and training sets.
# 
# The confusion matrix proves that this model to be very weak at identifying those clients who will default. 

# #### Let's check the coefficients, and check which variables are important and how they affect the process of loan approval

# In[11]:


# Printing the coefficients of logistic regression
cols = x.columns

coef_lg = lg.coef_

pd.DataFrame(coef_lg,columns = cols).T.sort_values(by = 0, ascending = False)


# In[12]:


odds = np.exp(lg.coef_[0]) # Finding the odds

# Adding the odds to a DataFrame and sorting the values
pd.DataFrame(odds, x_train.columns, columns = ['odds']).sort_values(by = 'odds', ascending = False) 


# **Insights**
# 1. Feature which positively affect the process of loan approval:
# 
# DEBTINC	0.106 
# DELINQ	0.056
# NINQ	0.040
# DEROG	0.036
# 
# odds:
# DEBTINC	1.11
# DELINQ	1.06
# NINQ	1.04
# DEROG	1.04
# 
# 
# 2. Features which negatively affect the process of loan approval:
# 
# CLAGE	-0.005
# YOJ	    -0.010
# CLNO	-0.012
# 
# odds:
# CLAGE	0.995
# YOJ	    0.990
# CLNO	0.988
# 
# 
# - An increase of 1 in a client's debt-to-income ratio (DEBTINC) increases a client's chance of defaulting by 11%.
# - An increase of 1 delinquent credit line (DELINQ) increases a client's chance of defauting by 6%.
# - An increase of 1 recent credit inquiries (NINQ) increases a client's chance of defaulting by 4%.
# - An increase of 1 major derogatory reports (DEROG) increases a client's chance of defaulting by 4%. 

# ### **Think about it:**
# - The above Logistic regression model was build on the threshold of 0.5, can we use different threshold? 
#     1. Yes, we can run a precision-recall curve to find where the optimal threshold is.
# 
# - How to get an optimal threshold and which curve will help you achieve?
#     1. We can find an optimal threshold through a precision-recall curve and use that to find where the optimal threshold is.
# 
# - How does, accuracy, precision and recall change on the threshold?
#     1. As Recall decreases, Precision increases. The acuracy is best represented by precision and recall at the threshold, as the individual measures of precision and recall are more accurate when the line is higher so to optimize both, the point where the two meet is chosen as a threshold.

# In[13]:


#The Precision-Recall Curve for Logistic Regression
y_scores_lg = lg.predict_proba(x_train) # predict_proba gives the probability of each observation belonging to each class


precisions_lg, recalls_lg, thresholds_lg = precision_recall_curve(y_train, y_scores_lg[:, 1])

# Plot values of precisions, recalls, and thresholds
plt.figure(figsize = (10, 7))

plt.plot(thresholds_lg, precisions_lg[:-1], 'b--', label = 'precision')

plt.plot(thresholds_lg, recalls_lg[:-1], 'g--', label = 'recall')

plt.xlabel('Threshold')

plt.legend(loc = 'upper left')

plt.ylim([0, 1])

plt.show()


# **Observation:**
# 
# 1. We can see that the precision and the recall are balanced for a threshold of about 0.35, not .5 like we tested earlier. This means that at .35 the accuracy is greatest as that is where precision and recall intersect.

# In[14]:


# performance of the model at .35 threshold
optimal_threshold1 = .35

y_pred_train = lg.predict_proba(x_train)

metrics_score(y_train, y_pred_train[:, 1] > optimal_threshold1)


# ### **Build a Decision Tree Model**

# ### **Think about it:**
# - In Logistic regression we treated the outliers and built the model, should we do the same for tree based models or not? If not, why?
#     1. For regression it is important to use the fully treated data set, for a decision tree model it may be better just to use data that has been flagged for missing values and treated with the mean, median, and mode. Decision trees are not sensitive to outliers or noise in the data, as they are not involved in the splitting of branches. 

# #### Data Preparation for the tree based model

# In[15]:


# Reading the raw data set
hm=pd.read_csv("C:/Users/Anne/OneDrive/Desktop/MIT/Capstone/hmeq.csv")

# Rename data set 
data2 = hm.copy()


# In[16]:


#For each column we create a binary flag for the row, if there is missing value in the row, then 1 else 0. 
def add_binary_flag(data2,col):
    '''
    df: It is the dataframe
    col: it is column which has missing values
    It returns a dataframe which has binary falg for missing values in column col
    '''
    new_col = str(col)
    new_col += '_missing_values_flag'
    data2[new_col] = data2[col].isna()
    return data2


# In[17]:


# Add binary flags
# List of columns that has missing values in it
missing_col = [col for col in data2.columns if data2[col].isnull().any()]

for colmn in missing_col:
    add_binary_flag(data2,colmn)
    


# In[18]:


#  Treat Missing values in numerical columns with median and mode in categorical variables
# Select numeric columns.
num_data = data2.select_dtypes('number')

# Select string and object columns.
cat_data = data2.select_dtypes('category').columns.tolist()#df.select_dtypes('object')

# Fill numeric columns with median.
data2[num_data.columns] = num_data.fillna(data2.median()) #fill NaN values in all columns with median

# Fill object columns with model.
for column in cat_data:
    mode = data2[column].mode()[0]
    data2[column] = data2[column].fillna(data2[column].mode()[0])


# #### Separating the target variable y and independent variable x

# In[19]:


# Drop dependent variable from dataframe and create the X(independent variable) matrix
x = data2.drop(columns = 'BAD') # all columns excluding BAD

# Create dummy variables for the categorical variables - Hint: use the get_dummies() function
x = pd.get_dummies(x, drop_first = True)

# Create y(dependent varibale)
y = data2['BAD']  # BAD is the dependent variable


# #### Split the data

# In[20]:


# Split the data into training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 1) 


# In[21]:


#Defining Decision tree model with class weights class_weight={0: 0.2, 1: 0.8}
dt = DecisionTreeClassifier(class_weight = {0: 0.2, 1: 0.8}, random_state = 1)


# In[22]:


#fitting Decision tree model
dt.fit(x_train, y_train)


# #### Checking the performance on the train dataset

# In[23]:


# Checking performance on the training data
y_train_pred_dt = dt.predict(x_train)

metrics_score(y_train, y_train_pred_dt)


# #### Checking the performance on the test dataset

# In[24]:


# Checking performance on the testing data
y_test_pred_dt = dt.predict(x_test)

metrics_score(y_test, y_test_pred_dt)


# **Insights**
# 1. The decision tree is overfitting on the test set as evidenced by demonstrating 100% score for all metrics. The model performs better on the training dataset as it is not overfitting. Importantly the recall score is 92% for casse 0 and 68% for case 1 (defaulters).

# ### **Think about it:**
# - Can we improve this model?
#     1. The first decision tree model can be improved as it was only using class_weights which was approximately the opposite of the imbalance in the original data.We can improve this model by reducing overfitting with hyperparameters. 
# - How to get optimal parameters in order to get the best possible results?
#     1. To get the best parameters for the best possible results we can use hyperparameter tuning on top of class_weights to test numerous different  parameters through Grid search.

# ### **Decision Tree - Hyperparameter Tuning**
# 
# * Hyperparameter tuning is tricky in the sense that **there is no direct way to calculate how a change in the hyperparameter value will reduce the loss of your model**, so we usually resort to experimentation. We'll use Grid search to perform hyperparameter tuning.
# * **Grid search is a tuning technique that attempts to compute the optimum values of hyperparameters.** 
# * **It is an exhaustive search** that is performed on the specific parameter values of a model.
# * The parameters of the estimator/model used to apply these methods are **optimized by cross-validated grid-search** over a parameter grid.
# 
# **Criterion {“gini”, “entropy”}**
# 
# The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
# 
# **max_depth** 
# 
# The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
# 
# **min_samples_leaf**
# 
# The minimum number of samples is required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.
# 
# You can learn about more Hyperpapameters on this link and try to tune them. 
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# 

# #### Using GridSearchCV for Hyperparameter tuning on the model

# In[25]:


# Choose the type of classifier. 
dt_tuned = DecisionTreeClassifier(random_state = 1, class_weight = {0: 0.2, 1: 0.8})


# Grid of parameters to choose from
parameters = {'max_depth': np.arange(2, 7), 
              'criterion': ['gini', 'entropy'],
              'min_samples_leaf': [5, 10, 20, 25]
             }


# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(recall_score, pos_label = 1)



# Run the grid search
gridCV = GridSearchCV(dt_tuned, parameters, scoring = scorer, cv = 10)


# Fit the GridSearch on train dataset
gridCV = gridCV.fit(x_train, y_train)


# Set the clf to the best combination of parameters
dt_tuned = gridCV.best_estimator_



# Fit the best algorithm to the data. 
dt_tuned.fit(x_train, y_train)


# #### Checking the performance on the train dataset

# In[26]:


# Checking performance on the training data based on the tuned model
y_train_pred_dt = dt_tuned.predict(x_train)

metrics_score(y_train, y_train_pred_dt)


# #### Checking the performance on the test dataset

# In[27]:


# Checking performance on the testing data based on the tuned model
y_test_pred_dt = dt_tuned.predict(x_test)

metrics_score(y_test, y_test_pred_dt)


# **Insights**
# 1. In comparison to the model with default values of hyperparameters, the performance on the training set has gone down. This makes sense because we are trying to reduce overfitting.
# 2. This model with the tuned hyperparameters demonstrates much better recall for case 1 (defaulters) when compared to the model with just class_weights.
# 3. This model is not overfitting the training data and giving very similar results on the test and training data sets.
# 4. The precision for this model is far below the minimum threshold of 65%. 
# 
# Recall for case 1 has gone up significantly from .63 to .78 in comparison to the previous model which means the tuned model will give a lower number of false negatives. However, the best model should have 85% accuracy, 70% recall and at-least 65% precision, so this model is not appropriate.

# #### Plotting the Decision Tree

# In[28]:


# Plot the decision  tree and analyze it to build the decision rule
features = list(x.columns)

plt.figure(figsize = (20, 20))

tree.plot_tree(dt_tuned, feature_names = features, filled = True, fontsize = 9, node_ids = True, class_names = True)

plt.show()


# #### Deduce the business rules apparent from the Decision Tree and write them down: 
# Note: Blue leaves represent the defaulting clients( BAD = 1), while the orange leaves represent the clients that repay (BAD = 0). Also, the more the number of observations in a leaf, the darker its color gets.
# 
# Observations:
# 1. The first split in the decision tree is one of the flagged values "DEBTINC_missing_values_flag" showing that the treated variable for DEBTINC missing values using the median and mode are one of the most decising factors in whether a client will default or not.
# 2. Clients who have a debt-to-income ratio more than .5 are more lkely to default.
# 3. Of those clients who have a debt-to-income ratio more than .5, the clients who had a number of delinquent credit lines over .5 were more likely to default. 
# 4. And of the clients who have a debt-to-income ratio over .5, have a number of delinquent credit lines over .5, those who have fewer major derogatory reports than .5, are the most likely clients to defualt.
# 

# ### **Building a Random Forest Classifier**
# 
# **Random Forest is a bagging algorithm where the base models are Decision Trees.** Samples are taken from the training data and on each sample a decision tree makes a prediction. 
# 
# **The results from all the decision trees are combined together and the final prediction is made using voting or averaging.**

# In[29]:


# Defining Random forest CLassifier
rf_estimator = RandomForestClassifier(random_state = 1, criterion = "entropy")

rf_estimator.fit(x_train,y_train)


# #### Checking the performance on the train dataset

# In[30]:


#Checking performance on the training data
y_pred_train3 = rf_estimator.predict(x_train)

metrics_score(y_train, y_pred_train3)


# #### Checking the performance on the test dataset

# In[31]:


# Checking performance on the test data
y_pred_test3 = rf_estimator.predict(x_test)

metrics_score(y_test, y_pred_test3)


# **Observations:**
# 1. The performance is not very good for this model. It is not the best for the company's needs as the recall and macro avg are quite low for class 1 which would increase false negatives and could result in the bank giving loans to people who will default.

# ### **Build a Random Forest model with Class Weights**

# In[32]:


# Defining Random Forest model with class weights class_weight={0: 0.2, 1: 0.8}
rf_estimator = RandomForestClassifier(random_state = 1, class_weight={0: 0.2, 1: 0.8})


# Fitting Random Forest model
rf_estimator.fit(x_train,y_train)


# #### Checking the performance on the train dataset

# In[33]:


# Checking performance on the train data
y_pred_train3 = rf_estimator.predict(x_train)

metrics_score(y_train, y_pred_train3)


# #### Checking the performance on the test dataset

# In[34]:


# Checking performance on the test data
y_pred_test3 = rf_estimator.predict(x_test)

metrics_score(y_test, y_pred_test3)


# ### **Think about it:**
# - Can we try different weights?
#     1. Yes, it is possible to change the class_weights as they decide how many instances of class 1 are developed for each instance of class 0. So should you want to represent one class more than the other you can change them accordingly. 
# - If yes, should we increase or decrease class weights for different classes? 
#     1. In this particular case, as most of the data represents class 0, it may be beneficial to increase the second weight, which increases how many instances of class 1 are generated per class 0.

# ### **Tuning the Random Forest**

# * Hyperparameter tuning is tricky in the sense that **there is no direct way to calculate how a change in the hyperparameter value will reduce the loss of your model**, so we usually resort to experimentation. We'll use Grid search to perform hyperparameter tuning.
# * **Grid search is a tuning technique that attempts to compute the optimum values of hyperparameters.** 
# * **It is an exhaustive search** that is performed on the specific parameter values of a model.
# * The parameters of the estimator/model used to apply these methods are **optimized by cross-validated grid-search** over a parameter grid.
# 
# 
# **n_estimators**: The number of trees in the forest.
# 
# **min_samples_split**: The minimum number of samples required to split an internal node:
# 
# **min_samples_leaf**: The minimum number of samples required to be at a leaf node. 
# 
# **max_features{“auto”, “sqrt”, “log2”, 'None'}**: The number of features to consider when looking for the best split.
# 
# - If “auto”, then max_features=sqrt(n_features).
# 
# - If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).
# 
# - If “log2”, then max_features=log2(n_features).
# 
# - If None, then max_features=n_features.
# 
# You can learn more about Random Forest Hyperparameters from the link given below and try to tune them
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

# #### **Warning:** This may take a long time depending on the parameters you tune. 

# In[35]:


# Choose the type of classifier. 
rf_estimator_tuned = RandomForestClassifier(criterion = "entropy", random_state = 1)



# Grid of parameters to choose from
parameters = {"n_estimators": [110, 120],
    "max_depth": [6, 7],
    "min_samples_leaf": [20, 25],
    "max_features": [0.8, 0.9],
    "max_samples": [0.9, 1],
    "class_weight": ["balanced",{0: 0.3, 1: 0.7}]
             }


# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(recall_score, pos_label = 1)


# Run the grid search on the training data using scorer=scorer and cv=5
grid_obj = GridSearchCV(rf_estimator_tuned, parameters, scoring = scorer, cv = 5)


#fit the GridSearch on train dataset
grid_obj = grid_obj.fit(x_train, y_train)



# Set the clf to the best combination of parameters
rf_estimator_tuned = grid_obj.best_estimator_


# Fit the best algorithm to the data. 
rf_estimator_tuned.fit(x_train, y_train)


# #### Checking the performance on the train dataset

# In[36]:


# Checking performance on the training data
y_pred_train5 = rf_estimator_tuned.predict(x_train)

metrics_score(y_train, y_pred_train5)


# #### Checking the performance on the test dataset

# In[37]:


# Checking performace on test dataset
y_pred_test5 = rf_estimator_tuned.predict(x_test)

metrics_score(y_test, y_pred_test5)


# **Insights:**
# 1. The tuned random forest gives slightly worse result for recall when compared to the random forest classifier with default parameters which is great as it shows the model is not overtuning.
# 2. The tuned model performs very consistently with a recall of 82% and 75% for class 1 of the test and training sets. These recall scores show that this could be a potential model for prediciting loan default among clients. 
# 
# The best model should have around 85% accuracy, 70% recall and at-least 65% precision, so this model meets all of the standards when using weighted avg which considers the proportion of each case in the data.

# #### Plot the Feature importance of the tuned Random Forest

# In[38]:


# importance of features in the tree building ( The importance of a feature is computed as the 
#(normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance )
# Checking performace on test dataset
importances = rf_estimator_tuned.feature_importances_

indices = np.argsort(importances)

feature_names = list(x.columns)

plt.figure(figsize = (12, 12))

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color = 'violet', align = 'center')

plt.yticks(range(len(indices)), [feature_names[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()


# ### **Think about it:**
# - We have only built 3 models so far, Logistic Regression, Decision Tree and Random Forest 
# - We can build other Machine Learning classification models like kNN, LDA, QDA or even Support Vector Machines (SVM).
# - Can we also perform feature engineering and create model features and build a more robust and accurate model for this problem statement? 
#     1. Similar to the benefit of adjusting class_weights, there could be a benefit in feature engineering as it would allow us to create a model that has more instances of class 1 that are defaulters. The creation of feature engineering may allow us to make a more accurate model as it will provide us with more data for potential supervised and unsurpervised learning.

# ### **Comparing Model Performances**

# In[39]:


def get_recall_score(model,flag=True,X_train=x_train,x_test=x_test):
    '''
    model : classifier to predict values of X

    '''
    a = [] # defining an empty list to store train and test results
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    train_recall = metrics.recall_score(y_train,pred_train)
    test_recall = metrics.recall_score(y_test,pred_test)
    a.append(train_recall) # adding train recall to list 
    a.append(test_recall) # adding test recall to list
    
    # If the flag is set to True then only the following print statements will be dispayed. The default value is set to True.
    if flag == True: 
        print("Recall on training set : ",metrics.recall_score(y_train,pred_train))
        print("Recall on test set : ",metrics.recall_score(y_test,pred_test))
    
    return a # returning the list with train and test scores


# In[40]:


##  Function to calculate precision score
def get_precision_score(model,flag=True,x_train=x_train,x_test=x_test):
    '''
    model : classifier to predict values of X

    '''
    b = []  # defining an empty list to store train and test results
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    train_precision = metrics.precision_score(y_train,pred_train)
    test_precision = metrics.precision_score(y_test,pred_test)
    b.append(train_precision) # adding train precision to list
    b.append(test_precision) # adding test precision to list
    
    # If the flag is set to True then only the following print statements will be dispayed. The default value is set to True.
    if flag == True: 
        print("Precision on training set : ",metrics.precision_score(y_train,pred_train))
        print("Precision on test set : ",metrics.precision_score(y_test,pred_test))

    return b # returning the list with train and test scores


# In[41]:


##  Function to calculate accuracy score
def get_accuracy_score(model,flag=True,x_train=x_train,x_test=x_test):
    '''
    model : classifier to predict values of X

    '''
    c = [] # defining an empty list to store train and test results
    train_acc = model.score(x_train,y_train)
    test_acc = model.score(x_test,y_test)
    c.append(train_acc) # adding train accuracy to list
    c.append(test_acc) # adding test accuracy to list
    
    # If the flag is set to True then only the following print statements will be dispayed. The default value is set to True.
    if flag == True:
        print("Accuracy on training set : ",model.score(x_train,y_train))
        print("Accuracy on test set : ",model.score(x_test,y_test))
    
    return c # returning the list with train and test scores


# In[42]:


# Make the list of all the model names 

models = [lg, dt, dt_tuned, rf_estimator, rf_estimator_tuned]
# defining empty lists to add train and test results
acc_train = []
acc_test = []
recall_train = []
recall_test = []
precision_train = []
precision_test = []

# looping through all the models to get the accuracy,recall and precision scores
for model in models:
     # accuracy score
    j = get_accuracy_score(model,False)
    acc_train.append(j[0])
    acc_test.append(j[1])

    # recall score
    k = get_recall_score(model,False)
    recall_train.append(k[0])
    recall_test.append(k[1])

    # precision score
    l = get_precision_score(model,False)
    precision_train.append(l[0])
    precision_test.append(l[1])


# In[43]:


# Mention the Model names in the list. for example 'Model': ['Decision Tree', 'Tuned Decision Tree'..... write tht names of all model built]
comparison_frame = pd.DataFrame({'Model':['Logistic Regression', 'Decision Tree', 'Tuned Decision Tree', 'Random Forest', 'Tuned Random Forest'], 
                                          'Train_Accuracy': acc_train,
                                          'Test_Accuracy': acc_test,
                                          'Train_Recall': recall_train,
                                          'Test_Recall': recall_test,
                                          'Train_Precision': precision_train,
                                          'Test_Precision': precision_test}) 
comparison_frame


# **Insights:**
# 1. The Tuned Decision Tree and Tuned Random Forest show the best recall on test data, and could be potential models for further loan-default analysis.
# 2. Both the Tuned Decsion Tree and the Tuned Random Forest dhow consistency in their test and training sets and they also do not show signs of overfitting

# **1. Refined insights -** What are the most meaningful insights from the data relevant to the problem?
# There are numerous important aspects of the data to understand in order to best predict which clients will default their loans.
# 
# 1. Based on 3 different means of testing the most important variables can be decided:
# *Coefficients and odds of the Logistic Regression Model-* showed that the most important features in determening whether a loan will default or not are "DEBTINC", "DELINQ", "NINQ", and "DEROG".
# *Plotted Decision Tree-* Showed that "DEBTINC_missing_values_flag", "DELINQ", and "DEROG_missing_values_flag", and "CLAGE" were the most deciding features in deciding whether a client will default or not.
# *Tuned Random Forest Feature Importance Graph-* Showed that "DEBTINC_missing_values_flag","DEBTINC", "DELINQ", and "CLAGE" were the most important features in deciding whether a client will default or not.
# 
# Considering these three different measures of feature importance, the most important features of this data set in deciding whether a client will default or not are: 1)"DEBTINC_missing_values_flag"/"DEBTINC", 2)"DELINQ", and 3)"DEROG".
# 
# 2. Particular to this problem of loan default, the best performance metric to use for the model is recall as recall reduces the instances of false negatives. If we predict that a client will not default but in reality they do, then the bank loses money due to that error, and that client could be a continued risk. Where if we predict that a client will default but in reality they do not, then the bank loses a potential customer. In this case it is more important to minimize the chance of the former so as not to lose large sums of money.
# 
# 3. It is also important to note that the data set is skewed and somewhat unbalanced, which means that the data set disproportionately represents clients who have not defaulted as compared to those who have, who happen to be the most important aspect of the problem. One effect of this is represented in the exploratory data analysis of variable histograms like  "YOJ" and "lOAN". This could be fixed with further techniques; If a data set is highly unbalanced you can use resampling on the training data or a SMOTE technique to double the sample size of the minority data (in our case defaulters). Our set is unbalanced such that clients who have repayed their loans make up the majority of the data and the data set may benefit from techniques like SMOTE.
# 

# **2. Comparison of various techniques and their relative performance -** How do different techniques perform? Which one is performing relatively better? Is there scope to improve the performance further?
# 
# Using the main performance metric of recall to judge performance, the efficacy of the five models can be compared: 
# 1. Logistic Regression - Recall for class 0 was overfit for both the test and train datasets, as demonstrated by the 100% rate in both. Recall for class 1 (the instance of defaulters) is incredibly low, almost 0 for both test and training sets, resembling failure as it has an exceptionally high number of false negatives. The version with a lower threshold of .35 still had exceptionally low recall (~20%)
# 2. Decision Tree (with weights) - The decision tree overfits on the test data as evidenced by demonstrating a 100% score for all metrics. The model performs better on the training dataset as it is not overfitting, the recall score is 92% for case 0 but is 68% for case 1 (defaulters). This model is likely still overfitting for case 0 as a recall score of 92% is quite high, and is performing below acceptable rates for our purposes for case 1.
# 3. Tuned Decision Tree - In comparison to the model with default values of hyperparameters the performance on the tuned tree decreased which is a good sign as we are trying to reduce overfitting. Recall for case 1 has gone up significantly from 63% to 78% in comparison to the model with just class_weights which means the tuned model will give a lower number of false negatives. This model is not overfitting the training data however it significantly underperforms in terms of precision, returning only 56% when the threshold requires precision > 65%.
# 4. Random Forest - The performance is not very good for this model as it is overfitting. It does not suit the company's needs at all as the recall is quite low for class 1 (69%) which would increase false negatives and could result in the bank giving loans to people who will default. The Random Forest with class_weights performs even worse than the basic parameter version, with a recall of(66%).
# 5. Tuned Random Forest - The tuned random forest gave slightly decreased results for recall when compared to the random forest classifier with default parameters which is good as it shows the model is not overfitting. The tuned model performed very consistently across all performance metrics and returned a recall of 82% and 75% for class 1 of the test and training sets. These recall scores show that this could be a potential model for prediciting loan default among clients. 
# 
# Of these five, the best performer is the **Tuned Random Forest** as it returns consistent resuts in both training and testing trails and its recall rate falls below the overfitting threshold but above an average perfomance. The Tuned Random Forest's performance shows that the model could be useful in further applications to unlearned datasets to predict loan defauting. The perfromance statistics are all above the threshold of 85% accuracy (TRF has 87%), 70% recall (TRF has 75%)  and at-least 65% precision (TRF has 68%), so this model meets and exceeds performance standards.
# 
# This model could be further improved by 1) adjusting class weights and finding the right balance for each model, 2) manipulating specific parameters withing hyperparameter tuning, and 3) using SMOTE or class_weights as a technique to treat the unbalanced and skewed data set.

# **3. Proposal for the final solution design -** What model do you propose to be adopted? Why is this the best solution to adopt?
# 1. In order to best predict which clients will default a loan, the bank should use the Tuned Random Forest model. 
# The Tuned Random Forest model returned 75% for recall on test data, and has exceeded the threshold for perfromance. This model is very consistent between testing and training sets and does not show signs of overfitting; it also has the potential to be imporoved for future datasets. 
