#!/usr/bin/env python
# coding: utf-8

# In[61]:

# importing libraries for data handling and analysis
from sklearn import svm, tree, linear_model, neighbors
from pandas import Series, DataFrame
from sklearn import naive_bayes, ensemble, discriminant_analysis, gaussian_process
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot, plot_mpl
from plotly.offline import iplot, init_notebook_mode
from sklearn.metrics import roc_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from dateutil.parser import parse
from time import time
from datetime import datetime
import string
import timeit
import sys
import re
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import auc, roc_auc_score, roc_curve, recall_score, log_loss
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn import metrics
from sklearn import model_selection
from sklearn import feature_selection
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import cufflinks
import cufflinks as cf
import plotly.graph_objs as go
import plotly.figure_factory as ff
import chart_studio.plotly as py
import plotly
from IPython.display import display
import pandas as pd
from pandas.plotting import scatter_matrix
from pandas import ExcelWriter
from pandas import ExcelFile
from openpyxl import load_workbook
from scipy.stats import norm, skew
from scipy import stats
import statsmodels.api as sm
import os
import numpy as np

# import js
# from js import document, window, XMLHttpRequest, p5

# importing libraries for data visualisations
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
color = sns.color_palette()
pd.options.display.max_columns = None

# Standard plotly imports

# Using plotly + cufflinks in offline mode
cf.set_config_file(offline=True)
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

# Sklearn modules for preprocessing
# from imblearn.over_sampling import SMOTE # SMOTE
# sklearn module for ML model selection
# import 'train_test_split'

# libraries for data modelling

# Common sklearn Model Helpers
# from sklearn.datasets import make_classification

# sklearn modules for performance metrics

# importing misceallenous libraries
# ip = get_ipython()
# ip.register_magics(jupyternotify.JupyterNotifyMagics)

# In[7]:


# pwd

# In[8]:

# pass csv file via input argument
fileinput = sys.argv[1]
if not ".csv" in fileinput:
    fileinput += ".csv"
hr = pd.read_csv(fileinput)

# add file directly to script
# hr = pd.read_csv('.\\resources\\HR_dataset.csv')


# In[9]:


# Headers
hr.head(5)


# In[10]:


# Columns
hr.shape


# In[11]:


hr.dtypes


# In[12]:


# checking for missing values


# In[13]:


hr.isnull().values.any()
# No missing Values


# In[14]:


# Breaking Down columns by their various types
hr.columns.to_series().groupby(hr.dtypes).groups


# In[15]:


# some statistics
hr.describe()


# In[16]:


# number of employees that stayed and left the company
hr['Attrition'].value_counts()


# In[17]:


# plotting histograms for the numerical columns or attributes
hr.hist(figsize=(26, 26))
plt.tight_layout()
plt.savefig('public\\images\\fig17.png', dpi=150)


# In[18]:


(mu, sigma) = norm.fit(hr.loc[hr['Attrition'] == 'Yes', 'Age'])
print(
    'Ex-exmployees: average age = {:.1f} years old and standard deviation = {:.1f}'.format(mu, sigma))
(mu, sigma) = norm.fit(hr.loc[hr['Attrition'] == 'No', 'Age'])
print('Current exmployees: average age = {:.1f} years old and standard deviation = {:.1f}'.format(
    mu, sigma))


# In[19]:


# visualisation
x1 = hr.loc[hr['Attrition'] == 'No', 'Age']
x2 = hr.loc[hr['Attrition'] == 'Yes', 'Age']
# Group data together
hist_data = [x1, x2]
group_labels = ['Active Employees', 'Ex-Employees']
# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels,
                         curve_type='kde', show_hist=False, show_rug=False)
# Add title
fig['layout'].update(title='Age Distribution in Percent by Attrition Status')
fig['layout'].update(xaxis=dict(range=[15, 60], dtick=5))
# Plot
plotly.offline.iplot(fig, filename='Distplot with Multiple Datasets')
#### plotly to matplotlib conversion using plot_mpl ####
plotly.io.write_image(fig, "public\\images\\fig19.png",
                      width=700, height=600, scale=2)


# In[20]:


# Education Field of employees
hr['EducationField'].value_counts()


# In[21]:


df_EducationField = pd.DataFrame(columns=["Field", "% of Leavers"])
i = 0
for field in list(hr['EducationField'].unique()):
    ratio = hr[(hr['EducationField'] == field) & (hr['Attrition'] ==
                                                  "Yes")].shape[0] / hr[hr['EducationField'] == field].shape[0]
    df_EducationField.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))
df_EF = df_EducationField.groupby(by="Field").sum()
df_EF.iplot(kind='bar', title='Leavers by Education Field (%)')


# In[22]:


# Gender of employees
hr['Gender'].value_counts()


# In[23]:


print("Normalised gender distribution of ex-employees in the dataset: Male = {:.1f}%; Female {:.1f}%.".format((hr[(hr['Attrition'] == 'Yes')
                                                                                                                  & (hr['Gender'] == 'Male')].shape[0] / hr[hr['Gender'] == 'Male'].shape[0])*100,
                                                                                                              (hr[(hr['Attrition'] == 'Yes') & (hr['Gender'] == 'Female')].shape[0] / hr[hr['Gender'] == 'Female'].shape[0])*100))

# In[24]:


df_Gender = pd.DataFrame(columns=["Gender", "% of Leavers"])
i = 0
for field in list(hr['Gender'].unique()):
    ratio = hr[(hr['Gender'] == field) & (hr['Attrition'] == "Yes")
               ].shape[0] / hr[hr['Gender'] == field].shape[0]
    df_Gender.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))
df_g = df_Gender.groupby(by="Gender").sum()
df_g.iplot(kind='bar', title='Leavers by Gender (%)')


# In[25]:


# Marital Status of employees
hr['MaritalStatus'].value_counts()


# In[26]:


df_Marital = pd.DataFrame(columns=["Marital Status", "% of Leavers"])
i = 0
for field in list(hr['MaritalStatus'].unique()):
    ratio = hr[(hr['MaritalStatus'] == field) & (hr['Attrition'] ==
                                                 "Yes")].shape[0] / hr[hr['MaritalStatus'] == field].shape[0]
    df_Marital.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))
df_M = df_Marital.groupby(by="Marital Status").sum()
df_M.iplot(kind='bar', title='Leavers by Marital Status (%)')


# In[27]:


# Distance from Home
print("Distance from home for employees to get to work is from {} to {} miles.".format(hr['DistanceFromHome'].min(),
                                                                                       hr['DistanceFromHome'].max()))


# In[28]:


print('Average distance from home for currently active employees: {:.2f} miles and ex-employees: {:.2f} miles'.format(
    hr[hr['Attrition'] == 'No']['DistanceFromHome'].mean(), hr[hr['Attrition'] == 'Yes']['DistanceFromHome'].mean()))


# In[29]:


# Add visualisation
x1 = hr.loc[hr['Attrition'] == 'No', 'DistanceFromHome']
x2 = hr.loc[hr['Attrition'] == 'Yes', 'DistanceFromHome']
# Group data together
hist_data = [x1, x2]
group_labels = ['Active Employees', 'Ex-Employees']
# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels,
                         curve_type='kde', show_hist=False, show_rug=False)
# Add title
fig['layout'].update(
    title='Distance From Home Distribution in Percent by Attrition Status')
fig['layout'].update(xaxis=dict(range=[0, 30], dtick=2))
# Plot
plotly.offline.iplot(fig, filename='Distplot with Multiple Datasets')
# Convert to matplotlib using plot_mpl #
plotly.io.write_image(fig, "public\\images\\fig29.png",
                      width=700, height=600, scale=2)


# In[25]:


# The organisation consists of several departments
hr['Department'].value_counts()


# In[30]:


df_Department = pd.DataFrame(columns=["Department", "% of Leavers"])
i = 0
for field in list(hr['Department'].unique()):
    ratio = hr[(hr['Department'] == field) & (hr['Attrition'] == "Yes")
               ].shape[0] / hr[hr['Department'] == field].shape[0]
    df_Department.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))
df_D = df_Department.groupby(by="Department").sum()
df_D.iplot(kind='bar', title='Leavers by Department (%)')


# In[27]:


# Employees have different business travel commitmnent depending on their roles and level in the organisation
hr['BusinessTravel'].value_counts()


# In[31]:


df_BusinessTravel = pd.DataFrame(columns=["Business Travel", "% of Leavers"])
i = 0
for field in list(hr['BusinessTravel'].unique()):
    ratio = hr[(hr['BusinessTravel'] == field) & (hr['Attrition'] ==
                                                  "Yes")].shape[0] / hr[hr['BusinessTravel'] == field].shape[0]
    df_BusinessTravel.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))
df_B = df_BusinessTravel.groupby(by="Business Travel").sum()
df_B.iplot(kind='bar', title='Leavers by Business Travel (%)')


# In[29]:


# Employees in the database have several roles on-file
hr['JobRole'].value_counts()


# In[32]:


df_JobRole = pd.DataFrame(columns=["Job Role", "% of Leavers"])
i = 0
for field in list(hr['JobRole'].unique()):
    ratio = hr[(hr['JobRole'] == field) & (hr['Attrition'] == "Yes")
               ].shape[0] / hr[hr['JobRole'] == field].shape[0]
    df_JobRole.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))
df_JB = df_JobRole.groupby(by="Job Role").sum()
df_JB.iplot(kind='bar', title='Leavers by Job Role (%)')


# In[31]:


hr['JobLevel'].value_counts()


# In[33]:


df_JobLevel = pd.DataFrame(columns=["Job Level", "% of Leavers"])
i = 0
for field in list(hr['JobLevel'].unique()):
    ratio = hr[(hr['JobLevel'] == field) & (hr['Attrition'] == "Yes")
               ].shape[0] / hr[hr['JobLevel'] == field].shape[0]
    df_JobLevel.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))
df_J = df_JobLevel.groupby(by="Job Level").sum()
df_J.iplot(kind='bar', title='Leavers by Job Level (%)')


# In[33]:


hr['JobInvolvement'].value_counts()


# In[34]:


df_JobInvolvement = pd.DataFrame(columns=["Job Involvement", "% of Leavers"])
i = 0
for field in list(hr['JobInvolvement'].unique()):
    ratio = hr[(hr['JobInvolvement'] == field) & (hr['Attrition'] ==
                                                  "Yes")].shape[0] / hr[hr['JobInvolvement'] == field].shape[0]
    df_JobInvolvement.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))
df_JV = df_JobInvolvement.groupby(by="Job Involvement").sum()
df_JV.iplot(kind='bar', title='Leavers by Job Involvement (%)')


# In[35]:


print("Number of training times last year varies from {} to {} years.".format(
    hr['TrainingTimesLastYear'].min(), hr['TrainingTimesLastYear'].max()))


# In[36]:


# Add visualisation
x1 = hr.loc[hr['Attrition'] == 'No', 'TrainingTimesLastYear']
x2 = hr.loc[hr['Attrition'] == 'Yes', 'TrainingTimesLastYear']
# Group data together
hist_data = [x1, x2]
group_labels = ['Active Employees', 'Ex-Employees']
# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels,
                         curve_type='kde', show_hist=False, show_rug=False)
# Add title
fig['layout'].update(
    title='Training Times Last Year metric in Percent by Attrition Status')
fig['layout'].update(xaxis=dict(range=[0, 6], dtick=1))
# Plot
plotly.offline.iplot(fig, filename='Distplot with Multiple Datasets')
# Convert to matplotlib using plot_mpl #
plotly.io.write_image(fig, "public\\images\\fig36.png",
                      width=700, height=600, scale=2)


# In[37]:


hr['NumCompaniesWorked'].value_counts()


# In[35]:


df_NumCompaniesWorked = pd.DataFrame(
    columns=["Num Companies Worked", "% of Leavers"])
i = 0
for field in list(hr['NumCompaniesWorked'].unique()):
    ratio = hr[(hr['NumCompaniesWorked'] == field) & (hr['Attrition'] ==
                                                      "Yes")].shape[0] / hr[hr['NumCompaniesWorked'] == field].shape[0]
    df_NumCompaniesWorked.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))
df_NW = df_NumCompaniesWorked.groupby(by="Num Companies Worked").sum()
df_NW.iplot(kind='bar', title='Leavers by Num Companies Worked (%)')


# In[39]:


# Number of years in company
print('Average Number of Years at the company for currently active employees: {:.2f} miles and ex-employees: {:.2f} years'.format(
    hr[hr['Attrition'] == 'No']['YearsAtCompany'].mean(), hr[hr['Attrition'] == 'Yes']['YearsAtCompany'].mean()))


# In[40]:


print("Number of Years at the company varies from {} to {} years.".format(
    hr['YearsAtCompany'].min(), hr['YearsAtCompany'].max()))


# In[41]:


# Add visualisation
x1 = hr.loc[hr['Attrition'] == 'No', 'YearsAtCompany']
x2 = hr.loc[hr['Attrition'] == 'Yes', 'YearsAtCompany']
# Group data together
hist_data = [x1, x2]
group_labels = ['Active Employees', 'Ex-Employees']
# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels,
                         curve_type='kde', show_hist=False, show_rug=False)
# Add title
fig['layout'].update(title='Years At Company in Percent by Attrition Status')
fig['layout'].update(xaxis=dict(range=[0, 40], dtick=5))
# Plot
plotly.offline.iplot(fig, filename='Distplot with Multiple Datasets')
# Convert to matplotlib using plot_mpl #
plotly.io.write_image(fig, "public\\images\\fig41.png",
                      width=700, height=600, scale=2)


# In[42]:


print("Number of Years in the current role varies from {} to {} years.".format(
    hr['YearsInCurrentRole'].min(), hr['YearsInCurrentRole'].max()))


# In[43]:


# Add visualisation
x1 = hr.loc[hr['Attrition'] == 'No', 'YearsInCurrentRole']
x2 = hr.loc[hr['Attrition'] == 'Yes', 'YearsInCurrentRole']
# Group data together
hist_data = [x1, x2]
group_labels = ['Active Employees', 'Ex-Employees']
# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels,
                         curve_type='kde', show_hist=False, show_rug=False)
# Add title
fig['layout'].update(
    title='Years InCurrent Role in Percent by Attrition Status')
fig['layout'].update(xaxis=dict(range=[0, 18], dtick=1))
# Plot
plotly.offline.iplot(fig, filename='Distplot with Multiple Datasets')
plotly.io.write_image(fig, "public\\images\\fig43.png",
                      width=700, height=600, scale=2)


# In[44]:


print("Number of Years since last promotion varies from {} to {} years.".format(
    hr['YearsSinceLastPromotion'].min(), hr['YearsSinceLastPromotion'].max()))


# In[45]:


# Add visualisation
x1 = hr.loc[hr['Attrition'] == 'No', 'YearsSinceLastPromotion']
x2 = hr.loc[hr['Attrition'] == 'Yes', 'YearsSinceLastPromotion']
# Group data together
hist_data = [x1, x2]
group_labels = ['Active Employees', 'Ex-Employees']
# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels,
                         curve_type='kde', show_hist=False, show_rug=False)
# Add title
fig['layout'].update(
    title='Years Since Last Promotion in Percent by Attrition Status')
fig['layout'].update(xaxis=dict(range=[0, 15], dtick=1))
# Plot
plotly.offline.iplot(fig, filename='Distplot with Multiple Datasets')
plotly.io.write_image(fig, "public\\images\\fig45.png",
                      width=700, height=600, scale=2)

# In[46]:


print("Total working years varies from {} to {} years.".format(
    hr['TotalWorkingYears'].min(), hr['TotalWorkingYears'].max()))


# In[47]:


# Add visualisation
x1 = hr.loc[hr['Attrition'] == 'No', 'TotalWorkingYears']
x2 = hr.loc[hr['Attrition'] == 'Yes', 'TotalWorkingYears']
# Group data together
hist_data = [x1, x2]
group_labels = ['Active Employees', 'Ex-Employees']
# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels,
                         curve_type='kde', show_hist=False, show_rug=False)
# Add title
fig['layout'].update(
    title='Total Working Years in Percent by Attrition Status')
fig['layout'].update(xaxis=dict(range=[0, 40], dtick=5))
# Plot
plotly.offline.iplot(fig, filename='Distplot with Multiple Datasets')
plotly.io.write_image(fig, "public\\images\\fig47.png",
                      width=700, height=600, scale=2)


# In[48]:


# Years working with current manager
print('Average Number of Years with current manager for currently active employees: {:.2f} miles and ex-employees: {:.2f} years'.format(
    hr[hr['Attrition'] == 'No']['YearsWithCurrManager'].mean(), hr[hr['Attrition'] == 'Yes']['YearsWithCurrManager'].mean()))


# In[49]:


print("Number of Years with current manager varies from {} to {} years.".format(
    hr['YearsWithCurrManager'].min(), hr['YearsWithCurrManager'].max()))


# In[50]:


# Add visualisation
x1 = hr.loc[hr['Attrition'] == 'No', 'YearsWithCurrManager']
x2 = hr.loc[hr['Attrition'] == 'Yes', 'YearsWithCurrManager']
# Group data together
hist_data = [x1, x2]
group_labels = ['Active Employees', 'Ex-Employees']
# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels,
                         curve_type='kde', show_hist=False, show_rug=False)
# Add title
fig['layout'].update(
    title='Years With Curr Manager in Percent by Attrition Status')
fig['layout'].update(xaxis=dict(range=[0, 17], dtick=1))
# Plot
plotly.offline.iplot(fig, filename='Distplot with Multiple Datasets')
plotly.io.write_image(fig, "public\\images\\fig50.png",
                      width=700, height=600, scale=2)


# In[51]:


# Work-Life Balance Score
hr['WorkLifeBalance'].value_counts()


# In[36]:


df_WorkLifeBalance = pd.DataFrame(columns=["WorkLifeBalance", "% of Leavers"])
i = 0
for field in list(hr['WorkLifeBalance'].unique()):
    ratio = hr[(hr['WorkLifeBalance'] == field) & (hr['Attrition'] ==
                                                   "Yes")].shape[0] / hr[hr['WorkLifeBalance'] == field].shape[0]
    df_WorkLifeBalance.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))
df_W = df_WorkLifeBalance.groupby(by="WorkLifeBalance").sum()
df_W.iplot(kind='bar', title='Leavers by WorkLifeBalance (%)')


# In[53]:


hr['StandardHours'].value_counts()


# In[54]:


hr['OverTime'].value_counts()


# In[37]:


df_OverTime = pd.DataFrame(columns=["OverTime", "% of Leavers"])
i = 0
for field in list(hr['OverTime'].unique()):
    ratio = hr[(hr['OverTime'] == field) & (hr['Attrition'] == "Yes")
               ].shape[0] / hr[hr['OverTime'] == field].shape[0]
    df_OverTime.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))
df_O = df_OverTime.groupby(by="OverTime").sum()
df_O.iplot(kind='bar', title='Leavers by OverTime (%)')


# In[56]:


# Pay/Salary Employee Information
print("Employee Hourly Rate varies from ${} to ${}.".format(
    hr['HourlyRate'].min(), hr['HourlyRate'].max()))


# In[57]:


print("Employee Daily Rate varies from ${} to ${}.".format(
    hr['DailyRate'].min(), hr['DailyRate'].max()))


# In[58]:


print("Employee Monthly Rate varies from ${} to ${}.".format(
    hr['MonthlyRate'].min(), hr['MonthlyRate'].max()))


# In[59]:


print("Employee Monthly Income varies from ${} to ${}.".format(
    hr['MonthlyIncome'].min(), hr['MonthlyIncome'].max()))


# In[60]:


# Add visualisation
x1 = hr.loc[hr['Attrition'] == 'No', 'MonthlyIncome']
x2 = hr.loc[hr['Attrition'] == 'Yes', 'MonthlyIncome']
# Group data together
hist_data = [x1, x2]
group_labels = ['Active Employees', 'Ex-Employees']
# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels,
                         curve_type='kde', show_hist=False, show_rug=False)
# Add title
fig['layout'].update(title='Monthly Income by Attrition Status')
fig['layout'].update(xaxis=dict(range=[0, 20000], dtick=2000))
# Plot
plotly.offline.iplot(fig, filename='Distplot with Multiple Datasets')
plotly.io.write_image(fig, "public\\images\\fig60.png",
                      width=700, height=600, scale=2)


# In[61]:


print("Percentage Salary Hikes varies from {}% to {}%.".format(
    hr['PercentSalaryHike'].min(), hr['PercentSalaryHike'].max()))


# In[62]:


# Add visualisation
x1 = hr.loc[hr['Attrition'] == 'No', 'PercentSalaryHike']
x2 = hr.loc[hr['Attrition'] == 'Yes', 'PercentSalaryHike']
# Group data together
hist_data = [x1, x2]
group_labels = ['Active Employees', 'Ex-Employees']
# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels,
                         curve_type='kde', show_hist=False, show_rug=False)
# Add title
fig['layout'].update(title='Percent Salary Hike by Attrition Status')
fig['layout'].update(xaxis=dict(range=[10, 26], dtick=1))
# Plot
plotly.offline.iplot(fig, filename='Distplot with Multiple Datasets')
plotly.io.write_image(fig, "public\\images\\fig62.png",
                      width=700, height=600, scale=2)

# In[63]:


print("Stock Option Levels varies from {} to {}.".format(
    hr['StockOptionLevel'].min(), hr['StockOptionLevel'].max()))


# In[64]:


print("Normalised percentage of leavers by Stock Option Level: 1: {:.2f}%, 2: {:.2f}%, 3: {:.2f}%".format(
    hr[(hr['Attrition'] == 'Yes') & (hr['StockOptionLevel'] == 1)
       ].shape[0] / hr[hr['StockOptionLevel'] == 1].shape[0]*100,
    hr[(hr['Attrition'] == 'Yes') & (hr['StockOptionLevel'] == 2)
       ].shape[0] / hr[hr['StockOptionLevel'] == 1].shape[0]*100,
    hr[(hr['Attrition'] == 'Yes') & (hr['StockOptionLevel'] == 3)].shape[0] / hr[hr['StockOptionLevel'] == 1].shape[0]*100))


# In[38]:


df_StockOptionLevel = pd.DataFrame(
    columns=["StockOptionLevel", "% of Leavers"])
i = 0
for field in list(hr['StockOptionLevel'].unique()):
    ratio = hr[(hr['StockOptionLevel'] == field) & (hr['Attrition'] ==
                                                    "Yes")].shape[0] / hr[hr['StockOptionLevel'] == field].shape[0]
    df_StockOptionLevel.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))
df_S = df_StockOptionLevel.groupby(by="StockOptionLevel").sum()
df_S.iplot(kind='bar', title='Leavers by Stock Option Level (%)')


# In[66]:


# Employee Satisfaction and Performance Information
# Environment Satisfaction was captured as: 1 'Low' 2 'Medium' 3 'High' 4 'Very High'.
# Proportion of Leaving Employees decreases as the Environment Satisfaction score increases.
hr['EnvironmentSatisfaction'].value_counts()


# In[39]:


df_EnvironmentSatisfaction = pd.DataFrame(
    columns=["EnvironmentSatisfaction", "% of Leavers"])
i = 0
for field in list(hr['EnvironmentSatisfaction'].unique()):
    ratio = hr[(hr['EnvironmentSatisfaction'] == field) & (hr['Attrition'] ==
                                                           "Yes")].shape[0] / hr[hr['EnvironmentSatisfaction'] == field].shape[0]
    df_EnvironmentSatisfaction.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))
df_E = df_EnvironmentSatisfaction.groupby(by="EnvironmentSatisfaction").sum()
df_E.iplot(kind='bar', title='Leavers by Environment Satisfaction (%)')


# In[68]:


# Job Satisfaction was captured as: 1 'Low' 2 'Medium' 3 'High' 4 'Very High'
hr['JobSatisfaction'].value_counts()


# In[40]:


df_JobSatisfaction = pd.DataFrame(columns=["JobSatisfaction", "% of Leavers"])
i = 0
for field in list(hr['JobSatisfaction'].unique()):
    ratio = hr[(hr['JobSatisfaction'] == field) & (hr['Attrition'] ==
                                                   "Yes")].shape[0] / hr[hr['JobSatisfaction'] == field].shape[0]
    df_JobSatisfaction.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))
df_JF = df_JobSatisfaction.groupby(by="JobSatisfaction").sum()
df_JF.iplot(kind='bar', title='Leavers by Job Satisfaction (%)')


# In[70]:


hr['RelationshipSatisfaction'].value_counts()


# In[41]:


df_RelationshipSatisfaction = pd.DataFrame(
    columns=["RelationshipSatisfaction", "% of Leavers"])
i = 0
for field in list(hr['RelationshipSatisfaction'].unique()):
    ratio = hr[(hr['RelationshipSatisfaction'] == field) & (hr['Attrition'] ==
                                                            "Yes")].shape[0] / hr[hr['RelationshipSatisfaction'] == field].shape[0]
    df_RelationshipSatisfaction.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))
df_R = df_RelationshipSatisfaction.groupby(by="RelationshipSatisfaction").sum()
df_R.iplot(kind='bar', title='Leavers by Relationship Satisfaction (%)')


# In[72]:


hr['PerformanceRating'].value_counts()


# In[73]:


print("Normalised percentage of leavers by Stock Option Level: 3: {:.2f}%, 4: {:.2f}%".format(
    hr[(hr['Attrition'] == 'Yes') & (hr['PerformanceRating'] == 3)
       ].shape[0] / hr[hr['StockOptionLevel'] == 1].shape[0]*100,
    hr[(hr['Attrition'] == 'Yes') & (hr['PerformanceRating'] == 4)].shape[0] / hr[hr['StockOptionLevel'] == 1].shape[0]*100))


# In[42]:


df_PerformanceRating = pd.DataFrame(
    columns=["PerformanceRating", "% of Leavers"])
i = 0
for field in list(hr['PerformanceRating'].unique()):
    ratio = hr[(hr['PerformanceRating'] == field) & (hr['Attrition'] ==
                                                     "Yes")].shape[0] / hr[hr['PerformanceRating'] == field].shape[0]
    df_PerformanceRating.loc[i] = (field, ratio*100)
    i += 1
#print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))
df_P = df_PerformanceRating.groupby(by="PerformanceRating").sum()
df_P.iplot(kind='bar', title='Leavers by Performance Rating (%)')


# In[75]:


# Target Variable: Attrition
# The feature 'Attrition' is what this Machine Learning problem is about. We are trying to predict the value of the feature
# 'Attrition' by using other related features associated with the employee's personal and professional history.
# Attrition indicates if the employee is currently active ('No') or has left the company ('Yes')
hr['Attrition'].value_counts()


# In[76]:


print("Percentage of Current Employees is {:.1f}% and of Ex-employees is: {:.1f}%".format(
    hr[hr['Attrition'] == 'No'].shape[0] / hr.shape[0]*100,
    hr[hr['Attrition'] == 'Yes'].shape[0] / hr.shape[0]*100))


# In[77]:


# Add visualisation
hr['Attrition'].iplot(kind='hist', xTitle='Attrition',
                      yTitle='count', title='Attrition Distribution')


# In[78]:


# As shown on the chart above, we see this is an imbalanced class problem.
# Indeed, the percentage of Current Employees in our dataset is 83.9% and the percentage of Ex-employees is: 16.1%
# Machine learning algorithms typically work best when the number of instances of each classes are roughly equal.
# We will have to address this target feature imbalance prior to implementing our Machine Learning algorithms.


# In[43]:


# Correlations (only linear corellations are measured)
# Find correlations with the target and sort
df_HR_TCOR = hr.copy()
df_HR_TCOR['Target'] = df_HR_TCOR['Attrition'].apply(
    lambda x: 0 if x == 'No' else 1)
df_HR_TCOR = df_HR_TCOR.drop(
    ['Attrition', 'EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18'], axis=1)
correlations = df_HR_TCOR.corr()['Target'].sort_values()
print('Most Positive Correlations: \n', correlations.tail(5))
print('\nMost Negative Correlations: \n', correlations.head(5))


# In[44]:


# plotting a heatmap to visualize the correlation between Attrition and these factors
# Calculate correlations
corr = df_HR_TCOR.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
# Heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(corr,
            vmax=.5,
            mask=mask,
            # annot=True, fmt='.2f',
            linewidths=.2, cmap="YlGnBu")
plt.tight_layout()
plt.savefig('public\\images\\fig44heatmap', dpi=150)

# In[81]:


# As shown above, "Monthly Rate", "Number of Companies Worked" and "Distance From Home" are positively correlated to Attrition;
# while "Total Working Years", "Job Level", and "Years In Current Role" are negatively correlated to Attrition.


# In[82]:


# before we move to the algorithm phase we need to remove useless columns or colums with only one unique value
hr = hr.drop('StandardHours', axis=1)
hr.head(2)
# dropped columns (employeecount,over18,employeenumber,standard hours)


# In[47]:


# Encoding
# Machine learning Models can only work with numerical attributes, so we have to convert the categorical
# labels with numeric values
# label emcoding and one-hot encoding will be used
En = LabelEncoder()
# Label Encoding will be used for columns with 2 or less unique values
En_count = 0
for col in hr.columns[1:]:
    if hr[col].dtype == 'object':
        if len(list(hr[col].unique())) <= 2:
            En.fit(hr[col])
            hr[col] = En.transform(hr[col])
            En_count += 1
print('{} columns were label encoded.'.format(En_count))


# In[48]:


# convert rest of categorical variable into dummy
hr = pd.get_dummies(hr, drop_first=True)
print(hr.shape)
hr.head(2)


# In[49]:


# Feature Scaling : this will shrink the range of values between o - n
# as Machine Learning algorithms perform better
# when input numerical variables fall within a similar scale
scaler = MinMaxScaler(feature_range=(0, 5))
hr_col = list(hr.columns)
hr_col.remove('Attrition')
for col in hr_col:
    hr[col] = hr[col].astype(float)
    hr[[col]] = scaler.fit_transform(hr[[col]])
hr['Attrition'] = pd.to_numeric(hr['Attrition'], downcast='float')
hr.head()


# In[50]:


# Splitting the data
# assign the target to a new dataframe and convert it to a numerical feature
target = hr['Attrition'].copy()


# In[51]:


type(target)


# In[52]:


# let's remove the target feature and redundant features from the dataset
hr.drop(['Attrition'], axis=1, inplace=True)
print('Size of Full dataset is: {}'.format(hr.shape))


# In[122]:


# Since we have class imbalance (i.e. more employees with turnover=0 than turnover=1)
# let's use stratify=y to maintain the same ratio as in the training dataset when splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(hr,
                                                    target,
                                                    test_size=0.25,
                                                    stratify=target)
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)


# In[180]:


# selection of algorithms to consider and set performance measure
models = []
models.append(('Logistic Regression', LogisticRegression(solver='liblinear',

                                                         class_weight='balanced')))
models.append(('Random Forest', RandomForestClassifier(
    n_estimators=110)))
models.append(('Gradient Boosting', GradientBoostingClassifier
               (n_estimators=20)))


# In[181]:


# evaluating each model in turn and provide accuracy and standard deviation scores
acc_R = []
auc_R = []
names = []
# set table to table to populate with performance results
col = ['Algorithm', 'ROC AUC Mean', 'ROC AUC STD',
       'Accuracy Mean', 'Accuracy STD']
df_R = pd.DataFrame(columns=col)
i = 0
# evaluate each model using cross-validation
for name, model in models:
    kfold = model_selection.KFold(
        n_splits=10)  # 10-fold cross-validation

    CV_ACC_R = model_selection.cross_val_score(  # accuracy scoring
        model, X_train, y_train, cv=kfold, scoring='accuracy')

    CV_ACC_R = model_selection.cross_val_score(  # roc_auc scoring
        model, X_train, y_train, cv=kfold, scoring='roc_auc')

    acc_R.append(CV_ACC_R)
    auc_R.append(CV_ACC_R)
    names.append(name)
    df_R.loc[i] = [name,
                   round(CV_ACC_R.mean()*100, 2),
                   round(CV_ACC_R.std()*100, 2),
                   round(CV_ACC_R.mean()*100, 2),
                   round(CV_ACC_R.std()*100, 2)
                   ]
    i += 1
df_R.sort_values(by=['ROC AUC Mean'], ascending=False)


# In[124]:


fig = plt.figure(figsize=(15, 7))
fig.suptitle('Algorithm Accuracy Comparison')
ax = fig.add_subplot(111)
plt.boxplot(acc_R)
ax.set_xticklabels(names)
plt.savefig('public\\algorithms\\A_fig124.png', dpi=150)


# In[125]:


# in this project there are not an equal number of observations in each class
# and all predictions and prediction errors are equally important this is why Classification Accuracy metric is not suitable enough
# let us try a different metric
fig = plt.figure(figsize=(15, 7))
fig.suptitle('Algorithm ROC AUC Comparison')
ax = fig.add_subplot(111)
plt.boxplot(auc_R)
ax.set_xticklabels(names)
plt.savefig('public\\algorithms\\A_fig125.png', dpi=150)


# In[ ]:


# we will be using both the logistic regression and the random forest


# In[128]:


# using 10 fold Cross-Validation to train our Logistic Regression Model and estimate its AUC score.
kfold = model_selection.KFold(n_splits=5)
modelCV = LogisticRegression(solver='liblinear',
                             class_weight="balanced")
scoring = 'roc_auc'
results = model_selection.cross_val_score(
    modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("AUC score (STD): %.2f (%.2f)" % (results.mean(), results.std()))


# In[129]:


# Fine tuning logic regession
# hyper-parameter list to fine-tune
param_grid = {'C': np.arange(1e-03, 2, 0.01)}
log_gs = GridSearchCV(LogisticRegression(solver='liblinear',  # setting GridSearchCV
                                         class_weight="balanced"),
                      iid=True,
                      return_train_score=True,
                      param_grid=param_grid,
                      scoring='roc_auc',
                      cv=10)

log_grid = log_gs.fit(X_train, y_train)
log_opt = log_grid.best_estimator_
results = log_gs.cv_results_

print('='*20)
print("best params: " + str(log_gs.best_estimator_))
print("best params: " + str(log_gs.best_params_))
print('best score:', log_gs.best_score_)
print('='*20)


# In[131]:


# Confusion Matrix
cnf_matrix = metrics.confusion_matrix(y_test, log_opt.predict(X_test))
class_names = [0, 1]  # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[133]:


print('Accuracy of Logistic Regression Classifier on test set: {:.2f}'.format(
    log_opt.score(X_test, y_test)*100))


# In[134]:


# Classification report for the optimised Log Regression
log_opt.fit(X_train, y_train)
print(classification_report(y_test, log_opt.predict(X_test)))


# In[187]:


log_opt.fit(X_train, y_train)  # fit optimised model to the training data
probs = log_opt.predict_proba(X_test)  # predict probabilities
# we will only keep probabilities associated with the employee leaving
probs = probs[:, 1]
# calculate AUC score using test dataset
logit_roc_auc = roc_auc_score(y_test, probs)
print('AUC score: %.3f' % logit_roc_auc)


# In[171]:


# Based on our ROC AUC comparison analysis, Logistic Regression and Random Forest show the highest mean AUC scores
# but we will use the random forest algorithm because
# it allows us to know which features are of the most importance in predicting the target feature ("attrition")
# fine-tuning the Random Forest algorithm's hyper-parameters by cross-validation against the AUC score
rf_classifier = RandomForestClassifier(class_weight="balanced")
param_grid = {'n_estimators': [110],
              'min_samples_split': [8],
              'min_samples_leaf': [1],
              'max_depth': [15]}

grid_obj = GridSearchCV(rf_classifier,
                        iid=True,
                        return_train_score=True,
                        param_grid=param_grid,
                        scoring='roc_auc',
                        cv=10)

grid_fit = grid_obj.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_

print('='*20)
print("best params: " + str(grid_obj.best_estimator_))
print("best params: " + str(grid_obj.best_params_))
print('best score:', grid_obj.best_score_)
print('='*20)


# In[155]:


# ploting the features by their importance.
importances = rf_opt.feature_importances_
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]
# Rearrange feature names so they match the sorted feature importances
names = [X_train.columns[i] for i in indices]
plt.figure(figsize=(15, 7))  # Create plot
plt.title("Feature Importance")  # Create plot title
plt.bar(range(X_train.shape[1]), importances[indices])  # Add bars
# Add feature names as x-axis labels
plt.xticks(range(X_train.shape[1]), names, rotation=90)
plt.tight_layout()
plt.savefig('public\\algorithms\\A_fig155.png', dpi=150)
# Show plot

# In[156]:


# coeficient
importances = rf_opt.feature_importances_
df_param_coeff = pd.DataFrame(columns=['Feature', 'Coefficient'])
for i in range(44):
    feat = X_train.columns[i]
    coeff = importances[i]
    df_param_coeff.loc[i] = (feat, coeff)
df_param_coeff.sort_values(by='Coefficient', ascending=False, inplace=True)
df_param_coeff = df_param_coeff.reset_index(drop=True)
df_param_coeff.head(10)


# In[157]:


# Confusion Matrix
cnf_matrix = metrics.confusion_matrix(y_test, rf_opt.predict(X_test))
class_names = [0, 1]  # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[173]:


# Accuracy
print('Accuracy of RandomForest Regression Classifier on test set: {:.2f}'.format(
    rf_opt.score(X_test, y_test)*100))


# In[174]:


# Classification report for the optimised RF Regression
rf_opt.fit(X_train, y_train)
print(classification_report(y_test, rf_opt.predict(X_test)))


# In[183]:


# checking auc score
rf_opt.fit(X_train, y_train)  # fit optimised model to the training data
probs = rf_opt.predict_proba(X_test)  # predict probabilities
# we will only keep probabilities associated with the employee leaving
probs = probs[:, 1]
# calculate AUC score using test dataset
rf_opt_roc_auc = roc_auc_score(y_test, probs)
print('AUC score: %.3f' % rf_opt_roc_auc)


# In[190]:


# Create ROC Graph
fpr, tpr, thresholds = roc_curve(y_test, log_opt.predict_proba(X_test)[:, 1])
rf_fpr, rf_tpr, rf_thresholds = roc_curve(
    y_test, rf_opt.predict_proba(X_test)[:, 1])
plt.figure(figsize=(14, 6))

# Plot Logistic Regression ROC
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.3f)' % logit_roc_auc)
# Plot Random Forest ROC
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.3f)' % rf_opt_roc_auc)
# Plot Base Rate ROC
plt.plot([0, 1], [0, 1], label='Base Rate' 'k--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('public\\algorithms\\A_fig190.png', dpi=150)


# In[98]:


# Trying the gradient boosting classifier fimding the best learning rate
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(
        n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2)
    gb_clf.fit(X_train, y_train)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training set): {0:.3f}".format(
        gb_clf.score(X_train, y_train)))
    print("Accuracy score (test set): {0:.3f}".format(
        gb_clf.score(X_test, y_test)))


# In[113]:


# fine tuning
gb_classifier = GradientBoostingClassifier()
param_grid = {'n_estimators': [20],
              'learning_rate': [0.75, 1],
              'max_features': [2],
              'max_depth': [2]}

grid_gb = GridSearchCV(gb_classifier,
                       iid=True,
                       return_train_score=True,
                       param_grid=param_grid,
                       scoring='roc_auc',
                       cv=10)

grid_fit = grid_gb.fit(X_train, y_train)
gb_opt = grid_fit.best_estimator_

print('='*20)
print("best params: " + str(grid_gb.best_estimator_))
print("best params: " + str(grid_gb.best_params_))
print('best score:', grid_gb.best_score_)
print('='*20)


# In[114]:

# Confusion Matrix
cnf_matrix = metrics.confusion_matrix(y_test, gb_opt.predict(X_test))
class_names = [0, 1]  # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.tight_layout()
# Extract Confusion Matrix Gradient boosting
plt.savefig('public\\images\\M_fig114.png', dpi=150)

# In[191]:


# Classification accuracy for gb
print('Accuracy of gradient boosting Classifier on test set: {:.2f}'.format(
    gb_opt.score(X_test, y_test)*100))

with open('file.txt', 'w+') as gb:
    print('Accuracy of gradient boosting Classifier on test set: {:.2f}'.format(
        gb_opt.score(X_test, y_test)*100), file=gb)
# In[116]:


gb_opt.fit(X_train, y_train)
print(classification_report(y_test, gb_opt.predict(X_test)))


# In[182]:


gb_opt.fit(X_train, y_train)  # fit optimised model to the training data
probs = gb_opt.predict_proba(X_test)  # predict probabilities
# we will only keep probabilities associated with the employee leaving
probs = probs[:, 1]
# calculate AUC score using test dataset
logit_roc_auc = roc_auc_score(y_test, probs)
print('AUC score: %.3f' % logit_roc_auc)

with open('file.txt', 'a+') as auc:
    print('AUC score: %.3f' % logit_roc_auc, file=auc)

# In[ ]:
