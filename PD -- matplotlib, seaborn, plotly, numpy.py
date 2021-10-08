#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df = pd.read_csv('/Users/maustin/Desktop/diabetes.csv')


# In[3]:


df.head()


# In[5]:


pip install pandas-profiling


# In[6]:


import pandas_profiling as pp
pp.ProfileReport(df)


# In[7]:


# INTERPRETATION OF RESULTS
# Variable: Pregnancies

# 17 distinct values
# 0 minimum pregnancies -- a person with 0 pregnancies
# 17 maximum pregnancies -- a person with 17 pregnancies
# Most records have between 3 and 4 pregnancies, closer to 4
# 14.5% of records have a zero value (0 pregnancies)
# 85.5% of records have a non-zero value (1-17 prengancies)


# In[11]:


pip install plotly


# In[12]:


# Visualization Imports

import matplotlib.pyplot as plt

import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.offline as py
py.init_notebook_mode(connected=True)

import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px
import numpy as np


# In[13]:


dist = df['Outcome'].value_counts()
colors = ['mediumturquoise', 'darkorange']
trace = go.Pie(values=(np.array(dist)),labels=dist.index)
layout = go.Layout(title='Diabetes Outcome')
data = [trace]
fig = go.Figure(trace,layout)
fig.update_traces(marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.show()


# In[15]:


# DIABETES OUTCOME

# 0 = No Diabetes
# 1 = Diabetes


# In[16]:


def df_to_plotly(df):
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist() }
import plotly.graph_objects as go
dfNew = df.corr()
fig = go.Figure(data=go.Heatmap(df_to_plotly(dfNew)))
fig.show()


# In[17]:


# POSITIVE CORRELATIONS

# Age and Pregnancies, z = 0.544
# Outcome and Glucose, z = 0.467
# Insulin and SkinThickenss, z = 0.438


# In[18]:


fig = px.scatter(df, x='Glucose', y='Insulin')
fig.update_traces(marker_color="turquoise",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Glucose and Insulin')
fig.show()


# In[19]:


fig = px.box(df, x='Outcome', y='Age')
fig.update_traces(marker_color="midnightblue",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Age and Outcome')
fig.show()


# In[20]:


plot = sns.boxplot(x='Outcome',y="BMI",data=df)

