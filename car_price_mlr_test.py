
# coding: utf-8

# In[62]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import seaborn as sns


# In[63]:


data=pd.read_csv('C:\\Users\KAVITA.LAPTOP-E2HD8FAQ\Contacts\Desktop\datasets\CarPrice_Assignment-1.csv')
data.head()


# In[64]:


data.describe()


# In[65]:


data.info()


# In[66]:


data.columns


# In[67]:


plt.figure(figsize=(20,10))
sns.heatmap(data.corr(),annot=True)


# In[68]:


data.drop(['wheelbase','carlength', 'carwidth', 'curbweight','enginesize','boreratio', 'citympg', 'highwaympg','enginelocation'], axis =1, inplace = True)


# In[69]:


data.columns


# In[70]:


company_name=data['CarName'].apply(lambda x : x.split(' ')[0])
data.insert(3,"Company_name",company_name)
data.drop("CarName" ,axis=1 , inplace=True)
data.head()


# In[71]:


data.Company_name=data.Company_name.str.lower()
def replace_name(a,b):
    data.Company_name.replace(a,b,inplace=True)
    
replace_name('maxda','mazda')
replace_name('porcshce','porsche')
replace_name('toyouta','toyota')
replace_name('vokswagen','volkswagen')
replace_name('vw','volkswagen')



# In[72]:


data.Company_name.unique()


# In[73]:


fig, ax = plt.subplots(figsize = (15,5))
plt1 = sns.countplot(data['Company_name'], order=pd.value_counts(data['Company_name']).index,)
plt1.set(xlabel = 'Company_name', ylabel= 'Count of Cars')
plt.xticks(rotation = 90)
plt.show()
plt.tight_layout()


# In[75]:


df_comp_avg_price = data[['Company_name','price']].groupby("Company_name", as_index = False).mean().rename(columns={'price':'brand_avg_price'})
plt1 = df_comp_avg_price.plot(x = 'Company_name', kind='bar',legend = False, sort_columns = True, figsize = (15,3))
plt1.set_xlabel("Company_name")
plt1.set_ylabel("Avg Price (Dollars)")
plt.xticks(rotation = 90)
plt.show()


# In[76]:


df_sym = pd.DataFrame(data['symboling'].value_counts())
df_sym.plot.pie(subplots=True,labels = df_sym.index.values, autopct='%1.1f%%', figsize = (15,7.5))
# Unsquish the pie.
plt.gca().set_aspect('equal')
plt.show()
plt.tight_layout()


# In[59]:


data['fueltype'] =data['fueltype'].map({'gas': 1, 'diesel': 0})
data['aspiration'] = data['aspiration'].map({'std': 1, 'turbo': 0})
data['doornumber'] = data['doornumber'].map({'two': 1, 'four': 0})

data.head()


# In[58]:


data=pd.get_dummies(data)
data.head()


# In[15]:


data.drop(["car_ID"],axis=1,inplace=True)


# In[16]:


data.head()


# In[17]:


cols_to_norm=["symboling","carheight","stroke","compressionratio","horsepower","peakrpm","price"]

normalised_data=data[cols_to_norm].apply(lambda x: (x-np.mean(x)/(max(x)-min(x))))
normalised_data.head()


# In[18]:


data.columns


# In[19]:


data['symboling'] = normalised_data['symboling']
data['carheight'] = normalised_data['carheight']
data['stroke'] = normalised_data['stroke']
data['price'] = normalised_data['price']
data['compressionratio'] = normalised_data['compressionratio']
data['horsepower'] = normalised_data['horsepower']
data['peakrpm']= normalised_data['peakrpm']
data.head()


# In[20]:


x=data[['symboling','fueltype', 'aspiration','doornumber', 'carheight', 'stroke', 'compressionratio','horsepower', 'peakrpm']]
y=data['price']


# In[21]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=47,test_size=0.25)
model=LinearRegression()
model.fit(x_train,y_train)


# In[22]:


a=float(input('symboling'))
b=float(input('fueltype'))
c=float(input('aspiration'))
d=float(input('doornumber'))
e=float(input('carheight'))
f=float(input('stroke'))
g=float(input('compressionratio'))
h=float(input('horsepower'))
i=float(input('peakrpm'))


# In[25]:


#x_tes=(a,b,c,d,e,f,g,h,i)
Y_Pred = model.predict(x_test)
Y_Pred


# In[26]:


x_test.shape


# In[27]:


y_test.shape


# In[28]:


Y_Pred.shape


# In[29]:


c = [i for i in range(1,53,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=3.5, linestyle="-")     #Plotting Actual
plt.plot(c,Y_Pred, color="red",  linewidth=3.5, linestyle="-")  #Plotting predicted
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Car Price', fontsize=16)


# In[33]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, Y_Pred)
r_squared = r2_score(y_test, Y_Pred)


# In[34]:


print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


# In[ ]:


import numpy as np
from sklearn import metrics
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_pred_m9)))


# In[40]:


# Error terms
c = [i for i in range(1,53,1)]
fig = plt.figure()
plt.plot(c,y_test-Y_Pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Car Price', fontsize=16)    

