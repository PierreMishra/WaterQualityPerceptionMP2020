#!/usr/bin/env python
# coding: utf-8

# Section 1: Data cleaning

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


df= pd.read_csv(
    'C:/Users/benha/Desktop/MasterProject/smileannotationsfinal.csv',
    names = ['id','text','category']
)
df.set_index('id', inplace=True)
df.category.value_counts()


# In[4]:


df = df[~df.category.str.contains('\|')]


# In[5]:


df = df[df.category != 'nocode']


# In[6]:


df.category.value_counts()


# In[7]:


possible_labels = df.category.unique()


# In[8]:


label_dict = {}
for index, possible_labels in enumerate(possible_labels):
    label_dict[possible_labels] = index


# In[9]:


df['label']=df.category.replace(label_dict)
df.head()


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


X_train, X_val, Y_train, Y_val = train_test_split(
    df.index.values,
    df.label.values,
    test_size=0.15, #15% for validation
    random_state=17,
    stratify=df.label.values
)


# In[12]:


df['data_type']= ['not_set']*df.shape[0]


# In[13]:


df.loc[X_train, 'data_type']='train'
df.loc[X_val, 'data_type']='val'
df.groupby(['category','label','data_type']).count()


# In[18]:


df.head()


# In[15]:


df.describe()


# Section 3: Exploring the data

# In[19]:


df.hist(bins=30, figsize=(13,5), color='r')


# In[23]:


sns.countplot(df['label'], label = 'Count')


# In[24]:


df['length'] = df['text'].apply(len)


# In[25]:


df


# In[27]:


df['length'].plot(bins=100, kind='hist')


# Section 4: Plot the wordcloud

# In[28]:


sentences = df['text'].tolist()


# In[29]:


sentences_as_one_string = " ".join(sentences)


# In[1]:


conda install -c conda-forge/label/cf201901 wordcloud


# In[2]:


from WordCloud import WordCloud


# In[ ]:




