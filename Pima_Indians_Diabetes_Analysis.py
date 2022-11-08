#!/usr/bin/env python
# coding: utf-8

# # Foundations of Data Science Project - Diabetes Analysis
# 
# ---------------
# ## Context
# ---------------
# 
# Diabetes is one of the most frequent diseases worldwide and the number of diabetic patients are growing over the years. The main cause of diabetes remains unknown, yet scientists believe that both genetic factors and environmental lifestyle play a major role in diabetes.
# 
# A few years ago research was done on a tribe in America which is called the Pima tribe (also known as the Pima Indians). In this tribe, it was found that the ladies are prone to diabetes very early. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients were females at least 21 years old of Pima Indian heritage. 
# 
# -----------------
# ## Objective
# -----------------
# 
# Here, we are analyzing different aspects of Diabetes in the Pima Indians tribe by doing Exploratory Data Analysis.
# 
# -------------------------
# ## Data Dictionary
# -------------------------
# 
# The dataset has the following information:
# 
# * Pregnancies: Number of times pregnant
# * Glucose: Plasma glucose concentration over 2 hours in an oral glucose tolerance test
# * BloodPressure: Diastolic blood pressure (mm Hg)
# * SkinThickness: Triceps skin fold thickness (mm)
# * Insulin: 2-Hour serum insulin (mu U/ml)
# * BMI: Body mass index (weight in kg/(height in m)^2)
# * DiabetesPedigreeFunction: A function that scores the likelihood of diabetes based on family history.
# * Age: Age in years
# * Outcome: Class variable (0: a person is not diabetic or 1: a person is diabetic)

# ## Q 1: Import the necessary libraries and briefly explain the use of each library (3 Marks)

# In[1]:


# Remove _____ & write the appropriate library name

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# #### Write your Answer here: 
numpy - used to perform mathematical operations on arrays, spans basic arithmetic to more complex alegbra.

pandas - used to manipulate series and dataframes. Can be used to clean and analyse data like through pivot tables.

seaborn - used for higher-level data vizualization, best for bivariate analysis and more complex graphs.

matplotlib.pyplot - used for more simple data vizuzalization, ".pyplot" is better for use in jupyter notebook

%matplotlib inline - used to connect backend of matplotlib to jupyter frontend so graphs and plots are displayed and stored in a saved notebook (not necessary if ".pyplot" is used).
# ## Q 2: Read the given dataset (2 Marks)

# In[2]:


# Remove _____ & write the appropriate function name

pima = pd.read_csv('C:/Users/Anne/OneDrive/Desktop/MIT/Week 2/diabetes.csv') #read csv from file path


# ## Q3. Show the last 10 records of the dataset. How many columns are there? (2 Marks)

# In[3]:


# Remove ______ and write the appropriate number in the function

pima.tail(10) # last 10 rows of the pima dataset


# #### Write your Answer here: 
# 
Ans 3:
There are 9 total columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, and Outcome. 
# ## Q4. Show the first 10 records of the dataset (2 Marks)

# In[4]:


# Remove _____ & write the appropriate function name and the number of rows to get in the output

pima.head(10) # first 10 rows of pima dataset


# ## Q5. What do you understand by the dimension of the dataset? Find the dimension of the `pima` dataframe. (3 Marks)

# In[5]:


# Remove _____ & write the appropriate function name

pima.shape # find the dimensions of the pima dataset: (rows,columns)


# #### Write your Answer here: 
# 
Ans 5:
.shape tells me that there are 768 rows and 9 columns in the dataset. 
# ## Q6. What do you understand by the size of the dataset? Find the size of the `pima` dataframe. (3 Marks)

# In[6]:


# Remove _____ & write the appropriate function name

pima.size # finds the total number of elements in the pima dataset


# #### Write your Answer here: 
# 
Ans 6:
.size tells me that the size of the dataframe is 6912 cells or elements. 
# ## Q7. What are the data types of all the variables in the data set? (2 Marks)
# **Hint: Use the info() function to get all the information about the dataset.**

# In[7]:


# Remove _____ & write the appropriate function name

pima.info() # returns information on the pima dataset by column including data types, null values, elements, and size 


# #### Write your Answer here: 
# 
Ans 7:
The data types in pima are float (BMI, DiabetesPedigreeFunction) and int (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, Age, Outcome).
# ## Q8. What do we mean by missing values? Are there any missing values in the `pima` dataframe? (4 Marks)

# In[8]:


# Remove _____ & write the appropriate function name

pima.isnull().values.any() # checks the pima dataset for any null values, returns boolean


# #### Write your Answer here: 
# 
Ans 8:
Missing values are cases where a respondent may not have given an answer, or a value was not recorded. This is depicted in the dataframe by NaN. We can search the dataframe using .isnull().values.any() to see if there are any values that show up as NaN in our dataframe. We can also use .notnull().values.any() to see if all of the elements are recorded values.
In this case .isnull returned "False", so there are no null, or NaN, values in our dataframe.
# ## Q9. What do the summary statistics of the data represent? Find the summary statistics for all variables except 'Outcome' in the `pima` data. Take one column/variable from the output table and explain all its statistical measures. (5 Marks)

# In[9]:


# Remove _____ & write the appropriate function name

pima.iloc[: , 0 : 8].describe() #using all rows locate indexes (columns) 0-7 and display their statistical summaries


# #### Write your Answer here: 
# 
Ans 9:
The summary statistics for the "Age" column represent the following:
count - there are 768 elements in the culumn "Age".
mean - the average/arithmetic mean age for the respondents is about 33 years old.
std - the standard deviation, the spread of respondent ages mostly varry by 11 years from the average of 33y (22-44y). 
min - the minimum age, the youngest respondent is 21.
25% - the 25th percentile, 25% of the respondents are less than 24.
50% - the 50th percentile, 50% of the respondents are less than 29.
75% - the 75th percentile, 75% of the respondents are less than 41.
max - the maximum age, the oldest respondent is 81.
# ## Q 10. Plot the distribution plot for the variable 'BloodPressure'. Write detailed observations from the plot. (2 Marks)

# In[10]:


# Remove _____ & write the appropriate library name

sns.displot(pima['BloodPressure'], kind = 'kde') # Seaborn kernel density estimate distribution plot on the variable 'BloodPressure'

plt.show() # displays displot


# #### Write your Answer here: 
# 
Ans 10:
The kde distribution plot for 'BloodPressure' represents a normal distribution trend with the majority (rougly 68%) of the blood pressure data falling between 60 and 85. It also displays the max and min blood pressure values though the more extreme blood pressure gets, the fewer points there are thus resulting in the bell curve weighted around 72.
# ## Q 11. What is the 'BMI' of the person having the highest 'Glucose'? (2 Marks)

# In[11]:


# Remove _____ & write the appropriate function name

pima[pima['Glucose'] == pima['Glucose'].max()]['BMI'] # In pima under glucose variable, locates max glucose level and that respondent's BMI


# #### Write your Answer here: 
# 
Ans 11:
The BMI of the person with the highest 'Glucose' is 42.9
# ## Q12.
# ### 12.1 What is the mean of the variable 'BMI'? 
# ### 12.2 What is the median of the variable 'BMI'? 
# ### 12.3 What is the mode of the variable 'BMI'?
# ### 12.4 Are the three measures of central tendency equal?
# 
# ### (4 Marks)

# In[12]:


# Remove _____ & write the appropriate function name

m1 = pima['BMI'].mean()  # defines m1 as the mean of BMI variable in pima dataset
print(m1)

m2 = pima['BMI'].median()  # defines m2 as the median of BMI variable in pima dataset
print(m2)

m3 = pima['BMI'].mode()[0]  # defines m3 as the mode of BMI variable in pima dataset
print(m3)


# #### Write your Answer here: 
# 
Ans 12:
1. The mean BMI is 32.45
2. The median BMI is 32.0
3. The mode BMI is 32.0
4 The measures for median and mode are the exact same, the mean is very similar but because mean summation includes outliers, the value is slightly higher than either the median or mode.
# ## Q13. How many women's 'Glucose' levels are above the mean level of 'Glucose'? (2 Marks)

# In[13]:


# Remove _____ & write the appropriate function name

pima[pima['Glucose'] > pima['Glucose'].mean()].shape[0] # finds number of entries in pima['Glucose'] that are above the mean of pima['Glucose']


# #### Write your Answer here: 
# 
Ans 13:
343 women have glucose levels above the mean glucose level of 72.25.
# ## Q14. How many women have their 'BloodPressure' equal to the median of 'BloodPressure' and their 'BMI' less than the median of 'BMI'? (2 Marks)

# In[14]:


# Remove _____ & write the appropriate column name

# prints a list of respondents that have pima['BloodPressure'] equal to the media of pima['BloodPressure'] AND a pima['BMI'] less than the median of pima['BMI']
# .count() counts the entries in each column and prints it next to the variable name
pima[(pima['BloodPressure'] == pima['BloodPressure'].median()) & (pima['BMI'] < pima['BMI'].median())].count()


# #### Write your Answer here: 
# 
Ans 14:
There are 22 women who have their blood pressure equal to the median of 'BloodPressure'(72) and a BMI less than the median of 'BMI'(32.0).
# ## Q15. Create a pairplot for the variables 'Glucose', 'SkinThickness', and 'DiabetesPedigreeFunction'. Write your observations from the plot. (3 Marks)

# In[15]:


# Remove _____ & write the appropriate function name

# seaborn pairplot of data from pima, compares 3 variables 'Glucose', 'SkinThickness', 'DiabetesPedigreeFunction', and a hue showing those with diabetes and those without
sns.pairplot(data = pima, vars = ['Glucose', 'SkinThickness', 'DiabetesPedigreeFunction'], hue = 'Outcome')
plt.show() #displays pairplot


# #### Write your Answer here: 
# 
Ans 15:
1. Glucose & SkinThickness - shows no distinct correlation between glucose and skin thickness.
2. Glucose & DiabetesPedigreeFunction - shows a very slight correlation (maybe r = 0.08) between glucose and DiabetesPedigreeFunction. 
3. DiabetesPedigreeFunction & Skin thickness - shows very slight positive correlation between DiabetesPedigreeFunction and skin thickness.

the hue adds in an observable trend. As glucose increases in all the graphs, the number of respondents with diabetes increases. 
# ## Q16. Plot the scatterplot between 'Glucose' and 'Insulin'. Write your observations from the plot. (4 Marks)

# In[16]:


# Remove _____ & write the appropriate function name
# seaborn scatterplot using data from pima dataset with x axis 'Glucose' and y axis 'Insulin' 
sns.scatterplot(x = 'Glucose', y = 'Insulin', data = pima)
plt.show() # displays plot


# #### Write your Answer here: 
# 
Ans 16:
There is a moderate positive correlation (probably r= 0.4) between 'Glucose' and 'Insulin'. As 'Glucose' levels increase, 'Insulin' also increases.
# ## Q 17. Plot the boxplot for the 'Age' variable. Are there outliers? (2 Marks)

# In[17]:


# Remove _____ & write the appropriate function and column name 

plt.boxplot(pima['Age']) # matplotlib boxplot using pima dataset and variable 'Age'

plt.title('Boxplot of Age') # creates title for boxplot above 'Boxplot of Age'
plt.ylabel('Age') # creates y-axis label for boxplot above 'Age'
plt.show() # displays boxplot 


# #### Write your Answer here: 
# 
Ans 17:
Yes, there are outliers. In a boxplot the upper and lower ticks represent upper and lower extremes, anything above or below those ticks are outliers. The 'Age' variable has around 7 outliers beyond the upper extreme.
# ## Q18. Plot histograms for the 'Age' variable to understand the number of women in different age groups given whether they have diabetes or not. Explain both histograms and compare them. (5 Marks)

# In[18]:


# Remove _____ & write the appropriate function and column name

# matplotlib histogram using data from pima where pima['outcome'] is 1 (has diabetes), splits into 5 bins and makes graph red
plt.hist(pima[pima['Outcome'] == 1]['Age'], bins = 5, color = 'r')
plt.title('Distribution of Age for Women who have Diabetes') # adds title to histogram
plt.xlabel('Age') # adds x-axis label to histogram 
plt.ylabel('Frequency') # adds y-axis label to histogram
plt.show() # displays histogram


# In[19]:


# Remove _____ & write the appropriate function and column name


# matplotlib histogram using data from pima where pima['outcome'] is 0 (does not have diabetes), and splits into 5 bins 
plt.hist(pima[pima['Outcome'] == 0]['Age'], bins = 5) 
plt.title('Distribution of Age for Women who do not have Diabetes') # adds title to graph 
plt.xlabel('Age') # adds x-axis label to graph
plt.ylabel('Frequency') # adds y-axis label to graph
plt.show() # displays histogram 


# #### Write your Answer here: 
# 
Ans 18:
1. The first histogram shows that of the respondents who have diabetes, most are under 50 years old. This seems to show the trend that mostly younger women have diabetes, but this is misleading. 
2. The second histogram shows that a huge majoirty of the people who do not have diabetes are under 50 or even 30 years old. 

When looked at together the graphs create a better picture, there may not be a corelation between age and diabetes, but there were far more young respondents in this dataset than older respondents which makes the graphs look skewed towards yonger women when observed independently.
# ## Q 19. What is the Interquartile Range of all the variables? Why is this used? Which plot visualizes the same? (5 Marks)

# In[20]:


# Remove _____ & write the appropriate variable name

Q1 = pima.quantile(0.25) # defines Q1 as the 25th percentile of all the variables of pima 
Q3 = pima.quantile(0.75) # defines Q3 as the 75th percentile of all the variables of pima

# print (Q3, Q1) to see Q3 and Q1 on top of one another to form range

IQR = Q3 - Q1 # subtracts the 25th percentile of pima variables from the 75th percentile of pima variables
print(IQR) # prints difference of Q3 minus Q1


# #### Write your Answer here: 
# 
Ans 19:
1.Variable (Q3 - Q1) = IQR
 Pregnancies (6 - 1)
 Glucose (140.2 - 99.7)
 BloodPressure (80 - 64)
 SkinThickness (32 - 20)
 Insulin (127.3 - 79.0)
 BMI (36.6 - 27.5)
 DiabetesPedigreeFunction (.63 - .24)
 Age(41 -24)
 Outcome (N/A)
 
2.The IQR is used to show spread in a dataset and represents the middle 50% of the data. 
3. A boxplot shows the IQR with a box, the top (or right) of the box representing the end of Q3 and the bottom (or left) of the box represents the end of Q1. 
# ## Q 20. Find and visualize the correlation matrix. Write your observations from the plot. (3 Marks)

# In[21]:


# Remove _____ & write the appropriate function name and run the code

corr_matrix = pima.iloc[ : ,0 : 8].corr() #finds Pearson correlation (r) for each of the variables excluding 'Outcome'

corr_matrix # prints the correlation matrix of all r values for each of the 8 columns 


# In[22]:


# Remove _____ & write the appropriate function name

plt.figure(figsize = (8, 8)) # sets matplotlib heatmap dimensions to 8 by 8

# creates seaborn heatmap using data from corr_matrix above, annot = true puts the data values in to each cell
sns.heatmap(corr_matrix, annot = True) 

# Display the plot
plt.show()


# #### Write your Answer here: 
# 
Ans 20:
variables with large correlation:
    Age & Preganancies (positive .54)
    BMI & Skin Thickness  (positive .53)
variables with medium correlation 
    Insulin & Glucose (positive .4)
    Age & Blood Pressure (positive .33)