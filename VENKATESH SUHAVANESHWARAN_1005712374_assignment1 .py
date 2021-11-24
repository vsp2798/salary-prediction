#!/usr/bin/env python
# coding: utf-8

#        #########################  SALARY PREDICTION  #############################

# STEP 1: CLEANINING OF DATA
# 
#  a) Finding all missing NAN values and imputing them with various methods like mode, mean and mode.
#  
#  b) converting categorical data to numerical data

# In[1]:


import pandas as pd
import random 
Salaries = pd.read_csv("multipleChoiceResponses.csv", low_memory = False)


# In[2]:


Salaries.dropna(subset=['Q9'],inplace=True) 


# In[3]:


Salaries.Q9.unique() 


# In[4]:


Salaries = Salaries[Salaries['Q9']!= "I do not wish to disclose my approximate yearly compensation"]


# In[5]:


Salaries.loc[1:,'index'] = Salaries[1:].reset_index().index


# In[6]:


Salaries.to_csv("Kaggle_Salary_clean.csv")


# In[7]:


df = pd.read_csv('Kaggle_Salary_clean.csv')
df.groupby('Q2').describe()


# Have removed 'OTHER_TEXT' columns as they are insignificant to the data set . 

# In[8]:


wot = [i for i in df.columns if i.endswith('_OTHER_TEXT')]
salary = df
for k in wot:
    salary = salary.drop(columns = [k])
salary.columns


# The number of columns have been decreased from 395 to 368 .

# In[9]:


salary.shape


# Dropped index column which is float value

# In[10]:


salary.drop(['index'], axis = 1)
salary.head()


# Exploring data frame: 

# Finding the columns which have NaN values. 

# In[11]:


salary.isnull().sum(axis = 0).iloc[0:50]
salary.iloc[:, :50]


# From observing the NaN values for the first 50 columns, it is clear that most of the NaN values are in multiple choice questions. Now we'll try to plot the same in the heat map .

# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize = (10,6))

sns.heatmap(salary.iloc[:,:50].isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# In[13]:


salary.isnull().sum(axis=0).iloc[:50]


# The yellow marks in the plot shows the NaN values. Hence it can be observed that most of the mutliple choice columns have NaN values. 

# REMOVING ROWS AND COLUMNS.

# Removing rows based on countries . Countries have been filtered out on the basis of number of people who have been surveyed. 
# 
# Below is the countplot of all the number of people who have been surveyed from each country.

# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,7))

chart = sns.countplot(x = 'Q3', data = salary)
type(chart)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90)


# It can be observed that most of the countries are outliers. Countries where number of people who have been surveyed is less than 500 have been filtered out as they are less than 10% when compared with countries like India and USA.

# In[15]:


usa = salary[salary['Q3'] == 'United States of America']
india = salary[salary['Q3'] == 'India']
china = salary[salary['Q3'] == 'China']
other = salary[salary['Q3'] == 'Other']
russia = salary[salary['Q3'] == 'Russia']
brazil = salary[salary['Q3'] == 'Brazil']
germany = salary[salary['Q3'] == 'Germany']

countries = pd.concat([usa,india,china,other,russia,brazil,germany])

countries.shape


# Now the number of rows have been decreased to 8904 from 15430 rows. 

# Since rows have been reduced down. Now we reduce the number of columns.

# Now we will look into columns which has only single choice answers.

# In[16]:


single_columns = countries[['Unnamed: 0',
 'Time from Start to Finish (seconds)',
 'Q1',
 'Q2',
 'Q3',
 'Q4',
 'Q5',
 'Q6',
 'Q7',
 'Q8',
 'Q9',
 'Q10','Q17','Q18','Q20','Q22',
 'Q23',
 'Q24',
 'Q25',
 'Q26', 'Q32','Q37','Q40','Q43','Q46','Q48','index']]

single_columns.head()
single_columns.isnull().sum()


# In[17]:


fig, ax = plt.subplots(figsize = (10,6))

sns.heatmap(single_columns.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# Can be observed that most of the NaN values are from Q17 to Q48.

# In[18]:




modeimputingcols = ['Q8','Q17','Q18','Q32','Q43']


# Now imputing mode and Nan values for the single choice columns

# For column Q8, since most of the people are inexperienced ( from 0 to 1 year experience ). The Null values in the columns are substitued by the mode of the columns which is 0-1 years.

# In[19]:


print(countries['Q8'].isnull().sum())


# In[20]:


plt.figure(figsize=(10,5))

colQ8 = sns.countplot(x = 'Q8', data = countries)
type(colQ8)
colQ8.set_xticklabels(colQ8.get_xticklabels(), rotation=90)


# likewise , let us look at Q17.

# In[21]:


plt.figure(figsize=(10,6))

colQ17 = sns.countplot(x = 'Q17', data = countries)
type(colQ17)
colQ17.set_xticklabels(colQ17.get_xticklabels(), rotation=90)


# For column Q18, since nearly 70% of the options chosen is python than any other programming language. Imputing the 45 NaN values ( which is harldy 0.5 % of the total number of values) would be a good fit than applying NaN values.  

# In[22]:


countries['Q18'].value_counts()


# In[23]:


plt.figure(figsize=(10,6))

colQ18 = sns.countplot(x = 'Q18', data = countries)
type(colQ18)
colQ18.set_xticklabels(colQ18.get_xticklabels(), rotation=90)


# In[24]:


#countries['Q32'].unique()
#countries['Q43'].unique()


# In[25]:


countries['Q8'].fillna(countries['Q8'].mode()[0],inplace=True)
countries['Q17'].fillna(countries['Q17'].mode()[0],inplace=True)
countries['Q18'].fillna(countries['Q18'].mode()[0],inplace=True)
countries['Q32'].fillna(countries['Q32'].mode()[0],inplace=True)
countries['Q43'].fillna(countries['Q43'].mode()[0],inplace=True)
countries['Q46'].fillna(countries['Q43'].mode()[0],inplace=True)


# In[26]:


countries.head()


# In[27]:


fig, ax = plt.subplots(figsize = (10,6))

sns.heatmap(countries.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# The above graphs shows the number of NAN values present in the dataset

# Now removing columns and justifying
# 
# Unnamed column has been removed as they are insignificant.
# 
# Q12_part_text have been removed as they are just random binary numbers. But Q12_multiple choice questions are selected.
# 
# Q16 and Q17 have similar type of questions. Hence Q16 has been removed.
# Removing Q16 reduces the number of columns.Also Q17 is more relevant because it tells us about how often the language is used and will be more helpful in predicting the salary. 
# 
# Now comparing Q19, Q20 and Q21. Q20 has more relevance to predict the salary. This is because the Q20 talks about the frequent usage of ML library rather than in the past few years. This is will be more related to skills in data science than Q19 and Q21.
# 
# Now moving on to Q31 and Q32. Both have similar questions. Q32 talks about the frequent usage of data in school or work. Deleting Q31 will result in reduction in columns and also the model will have less features to deal with. 

# In[28]:


s1 = countries.drop(['Unnamed: 0','Q12_Part_1_TEXT','Q12_Part_2_TEXT','Q12_Part_3_TEXT','Q12_Part_4_TEXT','Q12_Part_5_TEXT', 'Q16_Part_1','Q16_Part_2','Q16_Part_3','Q16_Part_4','Q16_Part_5','Q16_Part_6','Q16_Part_7','Q16_Part_8','Q16_Part_9','Q16_Part_10','Q16_Part_11','Q16_Part_12','Q16_Part_13','Q16_Part_14','Q16_Part_15','Q16_Part_16','Q16_Part_17','Q16_Part_18',
'Q19_Part_1','Q19_Part_2','Q19_Part_3','Q19_Part_4','Q19_Part_5','Q19_Part_6','Q19_Part_7','Q19_Part_8','Q19_Part_9','Q19_Part_10','Q19_Part_11','Q19_Part_12','Q19_Part_13','Q19_Part_14','Q19_Part_15','Q19_Part_16','Q19_Part_17','Q19_Part_18','Q19_Part_19','Q21_Part_1','Q21_Part_2','Q21_Part_3','Q21_Part_4','Q21_Part_5','Q21_Part_6','Q21_Part_7','Q21_Part_8','Q21_Part_9','Q21_Part_10','Q21_Part_11','Q21_Part_12','Q21_Part_13', 
'Q31_Part_1','Q31_Part_2','Q31_Part_3','Q31_Part_4','Q31_Part_5','Q31_Part_6','Q31_Part_7','Q31_Part_8','Q31_Part_9','Q31_Part_10','Q31_Part_11','Q31_Part_12'], axis = 1)


# q1617=[['Q17','Q16_Part_15']]#Q18 dropped earlier
# list(countries.iloc[0][q1617])

s1.shape


# Q34 and Q35 have similar questions . Either one of the columns can be removed as they are similar and do not add much to the accuracy of the model and only complexes the model. hence Q34 is deleted
# 
# Comparing Q36 and Q37. 
# Q36 tells us about the online platforms in which people began and Q37 tells us about how much time they have spent on various online platfroms . Q37 will be correlated to the skill set and salary prediction than Q36. Hence Q36 can be removed. Also decreasing the number of columns and complexity of the model.
# 
# Q38 has been deleted. Favourite media sources would not affect the salary predictions
# 

# In[29]:




s4 = s1.drop(['Q34_Part_1','Q34_Part_2','Q34_Part_3','Q34_Part_4','Q34_Part_5','Q34_Part_6','Q36_Part_1','Q36_Part_2','Q36_Part_3','Q36_Part_4','Q36_Part_5','Q36_Part_6','Q36_Part_7','Q36_Part_8','Q36_Part_9','Q36_Part_10','Q36_Part_11','Q36_Part_12','Q36_Part_13',
'Q38_Part_1','Q38_Part_2','Q38_Part_3','Q38_Part_4','Q38_Part_5','Q38_Part_6','Q38_Part_7','Q38_Part_8','Q38_Part_9','Q38_Part_10','Q38_Part_11','Q38_Part_12','Q38_Part_13','Q38_Part_14','Q38_Part_15','Q38_Part_16','Q38_Part_17','Q38_Part_18','Q38_Part_19','Q38_Part_20','Q38_Part_21','Q38_Part_22','Q49_Part_1','Q49_Part_2','Q49_Part_3','Q49_Part_4','Q49_Part_5','Q49_Part_6','Q49_Part_7','Q49_Part_8','Q49_Part_9','Q49_Part_10','Q49_Part_11','Q49_Part_12'], axis = 1)
s4.shape
s5 = s4.drop(['Q50_Part_1','Q50_Part_2','Q50_Part_3','Q50_Part_4','Q50_Part_5','Q50_Part_6','Q50_Part_7','Q50_Part_8','index'], axis = 1)
s5.shape
s6 = s5.drop(['Q32_OTHER','Q26','Q47_Part_1','Q47_Part_2','Q47_Part_3','Q47_Part_4','Q47_Part_5','Q47_Part_6','Q47_Part_7','Q47_Part_8','Q47_Part_9','Q47_Part_10','Q47_Part_11','Q47_Part_12','Q47_Part_13','Q47_Part_14','Q47_Part_15','Q47_Part_16'], axis = 1)
s6.shape


# ## ONE HOT ENCODING HAS BEEN DONE TO THE SINGLE ANSWER COLUMNS AND MULTIPLE CHOICE COLUMNS AFTER EXPLORATORY DATA ANALYSIS!!!!!

# ### EXPLORATORY DATA ANALYSIS

# TREND 1 : HISTOGRAM PLOT OF Job descriptions
# 
# It can be seen that most of the people who were surveyed were primarily dataScientists and Students. 

# In[30]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(30,7))

jobplot = sns.countplot(x = 'Q6', data = s6)
chart.set_xticklabels(jobplot.get_xticklabels(), rotation=90)


# Trend 2: Box plot has been plotted for the Gender and Salary columns . 

# STEP 1) NOTE : Have converted the salary column (categorical) to a numeric column (numerical) for plotting the graphs.

# In[31]:


def score_to_numeric(x):
    if x=='0-10,000':
        return 1
    if x=='10-20,000':
        return 2
    if x=='20-30,000':
        return 3
    if x=='30-40,000':
        return 4
    if x=='40-50,000':
        return 5
    if x=='50-60,000':
        return 6
    if x=='60-70,000':
        return 7
    if x=='70-80,000':
        return 8
    if x=='80-90,000':
        return 9
    if x=='90-100,000':
        return 10
    if x=='100-125,000':
        return 11
    if x=='125-150,000':
        return 12
    if x=='150-200,000':
        return 13
    if x=='200-250,000':
        return 14
    if x=='250-300,000':
        return 15
    if x=='300-400,000':
        return 16
    if x=='400-500,000':
        return 17
    if x=='500,000+':
        return 18


# In[32]:


s6['Q9_num'] = s6['Q9'].apply(score_to_numeric)


# In the box plot below, Salary column and Gender category has been interpreted.
# It can be observed that Female and Male have the same mean of salaries when compared with other people. The mean of salaries seems to be the highest for people who didn't prefer to expose their gender identity.

# In[33]:


sns.boxplot(x = 'Q1', y = 'Q9_num', data = s6)


# In[34]:


from wordcloud import WordCloud, STOPWORDS
import string
import matplotlib.pyplot as plt


# In[35]:


cloudcountry = salary['Q3']
comment_words = ' '
stopwords = set(STOPWORDS) 
  
for value in cloudcountry: 
      
    # typecaste each val to string 
    value = str(value) 
  
    # split the value 
    tokens = value.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
          
    for words in tokens: 
        comment_words = comment_words + words + ' '

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8)) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()        
  


# It can be seen that most of the people who were surveyed were primarily from USA and India. Not much of the people were been surveyed from countries like China , Russian and Germany. 

# #### FEATURE SELECTION - RANDOM FOREST CLASSIFIER

# Feature engineering is the process of transforming raw data into features that better represent the underlying problem to the predictive models, resulting in improved model accuracy on unseen data.
# 
# The feature selection technique is important because it helps in selecting only the important features of the columns and increases the accuracy of the model.
# 
# Random Forest feature selection technique have been used for selecting the features. 
# They provide good predictive performance , low overfitting and easy interpretability.
# 

# Dropping the categorical target column from the dataset and assigning to a dataset seperately.

# In[36]:


X = s6.drop(['Q9','Q9_num'], axis = 1)
y = s6['Q9']

X.shape


# Converting 'Time from Start to Finish (seconds)' column into a  numeric column from categorical column inorder to reduce the number of features or columns when encoded by get_dummies

# In[37]:


X['Time from Start to Finish (seconds)'] = pd.to_numeric(X['Time from Start to Finish (seconds)'])


# #### One hot encoding has been to all the multiple choice columns

# In[38]:



MCQ = ['Q11_Part_1','Q11_Part_2','Q11_Part_3','Q11_Part_4','Q11_Part_5','Q11_Part_6','Q11_Part_7'
,'Q13_Part_1','Q13_Part_2','Q13_Part_3','Q13_Part_4','Q13_Part_5','Q13_Part_6','Q13_Part_7','Q13_Part_8','Q13_Part_9','Q13_Part_10','Q13_Part_11','Q13_Part_12','Q13_Part_13','Q13_Part_14','Q13_Part_15','Q14_Part_1','Q14_Part_2','Q14_Part_3','Q14_Part_4','Q14_Part_5','Q14_Part_6','Q14_Part_7','Q14_Part_8','Q14_Part_9','Q14_Part_10','Q14_Part_11','Q15_Part_1','Q15_Part_2','Q15_Part_3','Q15_Part_4','Q15_Part_5','Q15_Part_6','Q15_Part_7',
'Q27_Part_1','Q27_Part_2','Q27_Part_3','Q27_Part_4','Q27_Part_5','Q27_Part_6','Q27_Part_7','Q27_Part_8','Q27_Part_9','Q27_Part_10','Q27_Part_11','Q27_Part_12','Q27_Part_13','Q27_Part_14','Q27_Part_15','Q27_Part_16','Q27_Part_17','Q27_Part_18','Q27_Part_19','Q27_Part_20','Q28_Part_1','Q28_Part_2','Q28_Part_3','Q28_Part_4','Q28_Part_5','Q28_Part_6','Q28_Part_7','Q28_Part_8','Q28_Part_9','Q28_Part_10',
'Q28_Part_11','Q28_Part_12','Q28_Part_13','Q28_Part_14','Q28_Part_15','Q28_Part_16','Q28_Part_17','Q28_Part_18','Q28_Part_19','Q28_Part_20','Q28_Part_21','Q28_Part_22','Q28_Part_23','Q28_Part_24','Q28_Part_25','Q28_Part_26','Q28_Part_27','Q28_Part_28','Q28_Part_29','Q28_Part_30','Q28_Part_31','Q28_Part_32','Q28_Part_33','Q28_Part_34','Q28_Part_35','Q28_Part_36','Q28_Part_37','Q28_Part_38','Q28_Part_39','Q28_Part_40','Q28_Part_41','Q28_Part_42','Q28_Part_43','Q29_Part_1','Q29_Part_2','Q29_Part_3','Q29_Part_4','Q29_Part_5','Q29_Part_6','Q29_Part_7','Q29_Part_8','Q29_Part_9','Q29_Part_10','Q29_Part_11','Q29_Part_12','Q29_Part_13','Q29_Part_14','Q29_Part_15','Q29_Part_16','Q29_Part_17','Q29_Part_18','Q29_Part_19','Q29_Part_20','Q29_Part_21','Q29_Part_22','Q29_Part_23','Q29_Part_24','Q29_Part_25','Q29_Part_26','Q29_Part_27','Q29_Part_28','Q30_Part_1','Q30_Part_2','Q30_Part_3','Q30_Part_4','Q30_Part_5','Q30_Part_6','Q30_Part_7','Q30_Part_8','Q30_Part_9','Q30_Part_10','Q30_Part_11','Q30_Part_12','Q30_Part_13','Q30_Part_14','Q30_Part_15','Q30_Part_16','Q30_Part_17','Q30_Part_18','Q30_Part_19','Q30_Part_20','Q30_Part_21','Q30_Part_22','Q30_Part_23','Q30_Part_24','Q30_Part_25','Q33_Part_1','Q33_Part_2','Q33_Part_3','Q33_Part_4','Q33_Part_5','Q33_Part_6','Q33_Part_7','Q33_Part_8','Q33_Part_9','Q33_Part_10','Q33_Part_11','Q35_Part_1','Q35_Part_2','Q35_Part_3','Q35_Part_4','Q35_Part_5','Q35_Part_6',
'Q39_Part_1','Q39_Part_2','Q41_Part_1','Q41_Part_2','Q41_Part_3','Q42_Part_1','Q42_Part_2','Q42_Part_3','Q42_Part_4','Q42_Part_5','Q44_Part_1','Q44_Part_2','Q44_Part_3','Q44_Part_4','Q44_Part_5','Q44_Part_6','Q45_Part_1','Q45_Part_2','Q45_Part_3','Q45_Part_4','Q45_Part_5','Q45_Part_6']



# In[39]:


MCQdf = X[MCQ].notnull().astype(int)


# In[40]:


MCQdf.shape


# #### Now encoding only the remaining single choice columns

# In[41]:


singledf = X[[i for i in X.columns if i not in MCQ]]
singledf.shape


# In[42]:


singledf1 = pd.get_dummies(singledf)

singledf1.head()


# In[43]:


concateddf = pd.concat([singledf1,MCQdf], axis =1)

type(concateddf)

y = pd.DataFrame(y)

type(y)

type(concateddf)


# #### Label encoding is done for the target column

# In[44]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y)
print(le.classes_)
y=le.transform(y).reshape(-1,1)
print(y)


# In[45]:


print(concateddf.shape)
print(y.shape)
type(y)


# In[46]:


type(y)


# #### Converting y again to a data frame
# 

# In[47]:



y = pd.DataFrame(y)
type(y)


# ### Now import random forest classifier

# In[48]:




import pandas as pd

from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


# In all feature selection procedures, it is a good practice to select the features by examining only the training set. This is to avoid overfitting. This is the reason behind splitting the data into training and test data. 

# In[49]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(concateddf, y, test_size=0.20, random_state=42)


print('Training set : ',X_train.shape,y_train.shape)

print('Testing set :',X_test.shape,y_test.shape)



# EXPLANATION OF WHY RANDOM CLASSIFIER WAS USED: from 'SelectFromModel',  it will select those features which importance is greater than the mean importance of all the features by default, but we can alter this threshold if we want.This is one of the main reasons behind choosing random forest feature selection technique.
# 
# How random forest are randomised : By selecting the data points used to build a tree and by selecting the features in each split test.
# 
# Benefits : They work well without heavy tuning of the parameters and dont require scaling of the data.

# In[50]:


sel = SelectFromModel(RandomForestClassifier())
sel.fit(X_train, y_train)


# sel.get_support():
# 
# It will return an array of boolean values. True for the features whose importance is greater than the mean importance and False for the rest

# In[51]:


sel.get_support()


# In[52]:


selected_feat= concateddf.columns[(sel.get_support())]
len(selected_feat)


# In[53]:


print(selected_feat)

slfeatures = selected_feat.values.tolist()

slfeatures


# In[54]:


concateddf = concateddf[slfeatures]
print(concateddf.shape)
print(y.shape)


# MODEL IMPLEMENTATION : LOGISTIC REGRESSION

# In[55]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
logmodel = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(concateddf, y, test_size=0.20, random_state=42)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# Now using K fold Classifer to check for accuracy
# 
# The default C value is 1.0 in this case.

# #### IMPORTANT NOTE: 
#  Now performing only in training data set. 
#  For cross-validation and hyperparameter the dataset used was Xtrain. This is done to make sure that the model is not trained on the test data. This aids in keeping the test data separate to make the model is generalized not overfitting to the training data.
#  

# In[56]:


import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

logmodel = LogisticRegression()
scaler = StandardScaler()
kfold = KFold(n_splits=10)
kfold.get_n_splits(X_train)

accuracy = np.zeros(10)
np_idx = 0

for train_idx, test_idx in kfold.split(X_train):
    X_train1, X_test1 = X_train.values[train_idx], X_train.values[test_idx]
    y_train1, y_test1 = y_train.values[train_idx], y_train.values[test_idx]
    
    X_train1 = scaler.fit_transform(X_train1)
    X_test1 = scaler.transform(X_test1)
    
    logmodel.fit(X_train1, y_train1)
    
    predictions = logmodel.predict(X_test1)
    prediction_prob = logmodel.predict_proba(X_test1)
    
    ACC = accuracy_score(predictions,y_test1)
    np_idx += 1
    
    print (ACC)
   

print ("Average Score: {}%({}%)".format(round(np.mean(ACC),3),round(np.std(ACC),3)))


# In[57]:


test_predictions = logmodel.predict(X_test)
accuracy_score(test_predictions,y_test)


# The accuracy seems to be very low in the test set 

# Plot the mean accuracy, the "learning curve", of the classifier on both the training and validation dataset.

# In[58]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
def plot_learning_curve(estimator, title, X_train, y_train, ylim=None, cv=None, n_jobs=1,                        train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy'):
    
    plt.figure(figsize=(10,6))
    plt.title(title)
    
    if ylim is not None:
        plt.ylim(*ylim)
        
    plt.xlabel("Training examples")
    plt.ylabel(scoring)
    
    train_sizes, train_scores, test_scores = learning_curve(estimator, X_train, y_train, cv=10, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,                     train_scores_mean + train_scores_std, alpha=0.1,                      color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    
    return plt


# In[59]:



plot_learning_curve(logmodel,'Logistic Regression', X_train, y_train, cv=10)


# ####  BIAS VARIANCE TRADE-OFF:
# 
# Plot learning curve has been plotted to know whether the model has high bias or high variance and how to improve the model performance. We can minimise the training error by decreasing the bias. And variance can be reduced by decreasing the difference between the training and testing error.
# 
# If the model has high bias, we should try adding more features which increases the model complexity and decrease the regularisation parameter lambda.
# 
# If the model has high variance as seen from above , we should try decreasing the set of features and increase the regularisation parameter lambda.
#     

# #### MODEL TUNING USING GRID SEARCH CV

# In[60]:


from sklearn.model_selection import GridSearchCV
# param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }



# clf1 = GridSearchCV(LogisticRegression(penalty='l2'), param_grid,cv=10)
# best_clf = clf1.fit(X_train, y_train)


# best_clf.best_params_

                            
param_grid = {'C': [0.1, 1, 0.5,0.6,0.7,0.8,0.9,2,3,4,5,6,7,8,9]}
clf= GridSearchCV(LogisticRegression(penalty='l2'), param_grid)
gridmodel=clf.fit(X_train, y_train)


# In[61]:


gridmodel.best_params_


# In[62]:


gridmodel.best_score_


# In[63]:


#Predict values based on new parameters
y_pred_acc = gridmodel.predict(X_test)
accuracy_score(y_pred_acc,y_test)


# 
# #### So now we change the default value of C to 0.1 as it is the best parameter observed among C values. Now we implement the model again and see the accuracy.
# 
# 

# MODEL IMPLEMENTATION 2:

# In[64]:


from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

logmodel1 = LogisticRegression( C = 0.1 )
scaler = StandardScaler()
kfold = KFold(n_splits=10)
kfold.get_n_splits(X_train)

accuracy = np.zeros(10)
np_idx = 0

for train_idx, test_idx in kfold.split(X_train):
    X_train1, X_test1 = X_train.values[train_idx], X_train.values[test_idx]
    y_train1, y_test1 = y_train.values[train_idx], y_train.values[test_idx]
    
    X_train1 = scaler.fit_transform(X_train1)
    X_test1 = scaler.transform(X_test1)
    
    logmodel1.fit(X_train1, y_train1)
    
    predictions = logmodel1.predict(X_test1)
    prediction_prob = logmodel1.predict_proba(X_test1)
    
    ACC = accuracy_score(predictions,y_test1)
    np_idx += 1
    
    print (ACC)
   

print ('Accuracy percentage on training data:',"Average Score: {}%({}%)".format(round(np.mean(ACC),3),round(np.std(ACC),3)))


# Accuracy percentage when perfomed on unseen test data

# In[65]:


test_predictions = logmodel1.predict(X_test)
print('Accuracy percentage:',(accuracy_score(test_predictions,y_test)*100), '%')


# The accuracy increased when using the optimal parameter of C =0.1
# 
# The optimal C obtained after tuning was 0.1. Since the C is over the lower side, this signigies that the model is penalized for the higher values of weights. By default C=1, after tuning the model accuracy improved for optimal C value of 0.1. Since, the weight of regulatization C was 0.6, the model was overfitting which was brought to optimal fit by hyperparameter tuning.

# Now the graph on the new optimised model with the optimum parameter C = 0.1 has been plotted. It can be seen that the difference between the training and cross-validation score is decreased a bit. Hence decreasing the variance.

# In[66]:


plot_learning_curve(logmodel1,'Logistic Regression', X_train, y_train, cv=10)


# In[ ]:





# #### Now implementing the optimal model with the test data.

# In[67]:


from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

logmodel1 = LogisticRegression( C = 0.1 )
scaler = StandardScaler()
kfold = KFold(n_splits=10)
kfold.get_n_splits(X_test)

accuracy = np.zeros(10)
np_idx = 0

for train_idx, test_idx in kfold.split(X_test):
    X_train2, X_test2 = X_test.values[train_idx], X_test.values[test_idx]
    y_train2, y_test2 = y_test.values[train_idx], y_test.values[test_idx]
    
    X_train2 = scaler.fit_transform(X_train2)
    X_test2 = scaler.transform(X_test2)
    
    logmodel1.fit(X_train2, y_train2)
    
    predictions = logmodel1.predict(X_test2)
    prediction_prob1 = logmodel1.predict_proba(X_test2)
    
    ACC = accuracy_score(predictions,y_test2)
    np_idx += 1
    
    print (ACC)
   

print ('Accuracy percentage on training data:',"Average Score: {}%({}%)".format(round(np.mean(ACC),3),round(np.std(ACC),3)))


# #### TESTING AND DISCUSSION:
# 
# The model perfroms bad in test set and has decreased its percentage of accuracy but not much . So the model is a normal fit. Not underfit or overfit.

# The probability for both training and testing data is plotted below using a bar plot

# In[68]:


print(prediction_prob)
print(prediction_prob1)


# In[69]:


columns = ['0-10,000', '10-20,000', '20-30,000', '30-40,000', '40-50,000', '50-60,000','60-70,000', '70-80,000', '80-90,000', '90-100,000', '100-125,000', '125-150,000', '150-200,000','200-250,000', '250-300,000', '300-400,000','400-500,000', '500,000+' ]


# The plots have been plotted for person 2 on the dataset on both the training and testing datasets

# In[70]:


training_plot = pd.DataFrame(prediction_prob, columns = columns)
import matplotlib.pyplot as plt

plt.figure(figsize = (10,6))
plt.bar(columns, training_plot.loc[1])
plt.xlabel('salary')
plt.ylabel('probabality')
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  )
plt.title('training plot')
plt.show()



# In[71]:


testing_plot = pd.DataFrame(prediction_prob1, columns = columns)

plt.figure(figsize = (10,6))
plt.bar(columns, testing_plot.loc[1])
plt.xlabel('salary')
plt.ylabel('probabality')
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)
plt.title('testing plot')
plt.show()

