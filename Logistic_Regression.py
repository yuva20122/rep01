#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')


# In[41]:


data = pd.read_csv('diabetes.csv')


# In[42]:


data.head()


# In[43]:


data.shape


# In[44]:


data.describe()


# In[45]:


#Check data distribution of each columns
plt.figure(figsize = (20,25),facecolor ='yellow')
plotnumber =1
for column in data:
    if plotnumber <=9:
        ax = plt.subplot(3,3,plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column,fontsize = 20)
    plotnumber +=1
plt.tight_layout()


# In[46]:


#Replacing zero values with mean of the column

data.head()


# In[47]:


data['BMI']=data['BMI'].replace(0,data['BMI'].mean())
data['BloodPressure']=data['BloodPressure'].replace(0,data['BloodPressure'].mean())
data['Glucose']=data['Glucose'].replace(0,data['Glucose'].mean())
data['Insulin']=data['Insulin'].replace(0,data['Insulin'].mean())
data['SkinThickness']=data['SkinThickness'].replace(0,data['SkinThickness'].mean())


# In[48]:


plt.figure(figsize = (20,25),facecolor ='yellow')
plotnumber =1
for column in data:
    if plotnumber <=9:
        ax = plt.subplot(3,3,plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column,fontsize = 20)
    plotnumber +=1
plt.tight_layout()


# In[49]:


df_features=data.drop('Outcome',axis=1)


# In[50]:


plt.figure(figsize = (20,25),)
graph =1
for column in df_features:
    if graph <=8:
        ax = plt.subplot(3,3,graph)
        sns.boxplot(data=df_features[column])
        plt.xlabel(column,fontsize = 20)
    graph +=1
plt.tight_layout()


# In[51]:


data.shape


# In[52]:


#Find Inter Quantile Range to find outlilers
#1st quantile
q1 = data.quantile(0.25)
#3rd quantile
q3 = data.quantile(0.75)
#IQR
iqr = q3 - q1


# In[53]:


# Outlier detection formula
# higher side >> q3 + 1.5 * iqr
# lower side >> q1 - 1.5 * iqr


# In[54]:


data.head()


# In[55]:


#validating one outlier
preg_high = (q3.Pregnancies + (1.5 * iqr.Pregnancies))
preg_high


# In[56]:


outli_index =np.where(data['Pregnancies'] > preg_high )
outli_index


# In[57]:


data = data.drop(data.index[outli_index])
data.shape


# In[58]:


data.reset_index()


# In[59]:


BP_high = (q3.BloodPressure + (1.5 * iqr.BloodPressure))
outliBP_index =np.where(data['BloodPressure'] > BP_high )
data = data.drop(data.index[outliBP_index])
print(data.shape)
data.reset_index()


# In[60]:


ST_high = (q3.SkinThickness + (1.5 * iqr.SkinThickness))
outliST_index =np.where(data['SkinThickness'] > ST_high )
data = data.drop(data.index[outliST_index])
print(data.shape)
data.reset_index()


# In[61]:


IN_high = (q3.Insulin + (1.5 * iqr.Insulin))
outliIN_index =np.where(data['Insulin'] > IN_high )
data = data.drop(data.index[outliIN_index])
print(data.shape)
data.reset_index()


# In[62]:


DPF_high = (q3.DiabetesPedigreeFunction + (1.5 * iqr.DiabetesPedigreeFunction))
outliDPF_index =np.where(data['DiabetesPedigreeFunction'] > DPF_high )
data = data.drop(data.index[outliDPF_index])
print(data.shape)
data.reset_index()


# In[63]:


Age_high = (q3.Age + (1.5 * iqr.Age))
outliAge_index =np.where(data['Age'] > Age_high )
data = data.drop(data.index[outliAge_index])
print(data.shape)
data.reset_index()


# In[64]:


BP_low = (q1.BloodPressure - (1.5 * iqr.BloodPressure))
outliBPl_index =np.where(data['BloodPressure'] < BP_low )
data = data.drop(data.index[outliBPl_index])
print(data.shape)
data.reset_index()


# In[65]:


BMI_high = (q3.BMI + (1.5 * iqr.BMI))
outliBMI_index =np.where(data['BMI'] > BMI_high )
data = data.drop(data.index[outliBMI_index])
print(data.shape)
data.reset_index()


# In[66]:


plt.figure(figsize = (20,25),facecolor ='white')
plotnumber =1
for column in data:
    if plotnumber <=9:
        ax = plt.subplot(3,3,plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column,fontsize = 20)
    plotnumber +=1
plt.tight_layout()


# In[67]:


#find relation feature Vs feature


# In[68]:


x= data.drop(columns=['Outcome'])
y= data['Outcome']


# In[69]:


#Lets us see how features are related to class
plt.figure(figsize = (15,20))
number =1
for column in x:
    if number <=9:
        ax = plt.subplot(3,3,number)
        sns.stripplot(y,x[column])
        plt.xlabel(column,fontsize = 20)
        number +=1
plt.tight_layout()


# In[70]:


#Check multi collinearity 
scalar = StandardScaler()
X_scaled = scalar.fit_transform(x)


# In[71]:


X_scaled.shape[1] # collinearity checcked here for 8 differnt columns


# In[72]:


#finding variance inflation factor
vif = pd.DataFrame()
vif["vif"]=[variance_inflation_factor(X_scaled,i)for i in range(X_scaled.shape[1])]
vif["Features"]=x.columns
vif


# In[73]:


# In above case we can observe that variance inflation factor is less than 5 hence we can conclude that there is 
# collinariety ie there is no relation between features. vif factor consideration in case is 5 is only consider this case only it can be change 


# In[74]:


x_train,x_test,y_train,y_test = train_test_split(X_scaled,y,test_size =0.25,random_state= 348)


# In[75]:


log_reg= LogisticRegression()


# In[76]:


log_reg.fit(x_train,y_train)


# In[77]:


y_predic = log_reg.predict(x_test)
y_predic


# In[39]:


accuracy = accuracy_score(y_test,y_predic)
accuracy


# In[79]:


con_mat = confusion_matrix(y_test,y_predic)
con_mat


# In[80]:


from sklearn.metrics import classification_report


# In[81]:


print(classification_report(y_test,y_predic))


# In[82]:


#ROC curve
fpr,tpr,thresholds = roc_curve(y_test,y_predic)


# In[83]:


#thresholds should be read from zero
print('False Positive Rate-',fpr)
print('True Positive Rate-',tpr)
print('Thresholds-',thresholds)


# In[85]:


plt.plot(fpr, tpr, color ='orange',label = 'ROC')
plt.plot([0,1],[0,1],color = 'darkblue', linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristics ROC Curve')
plt.legend()
plt.show()


# In[86]:


#How much area under curve
auc_curve = roc_auc_score(y_test,y_predic)
print(auc_curve)


# In[ ]:




