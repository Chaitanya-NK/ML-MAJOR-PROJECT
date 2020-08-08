#!/usr/bin/env python
# coding: utf-8

# Problem Statement: You will be predicting the costs of used cars given the data collected from various sources and distributed across various locations in India.
# 
# FEATURES:
# 
# Name: The brand and model of the car.
# 
# Location: The location in which the car is being sold or is available for purchase.
# 
# Year: The year or edition of the model.
# 
# Kilometers_Driven: The total kilometres driven in the car by the previous owner(s) in KM.
# 
# Fuel_Type: The type of fuel used by the car.
# 
# Transmission: The type of transmission used by the car.
# 
# Owner_Type: Whether the ownership is Firsthand, Second hand or other.
# 
# Mileage: The standard mileage offered by the car company in kmpl or km/kg
# 
# Engine: The displacement volume of the engine in cc.
# 
# Power: The maximum power of the engine in bhp.
# 
# Seats: The number of seats in the car.
# 
# Price: The price of the used car in INR Lakhs.
# 
# Tasks:
# 
# 1.Clean Data(Null value removal, Outlier identification)
# 
# 2.Null Values(Dropping the rows /Columns and what is the reason or how you are imputing the null).
# 
# 3.EDA(Minor Project to understand the relations, repeat the same here)
# 
# 4.Handle Categorical Variable(Using Label Encoding/One hot encoding)
# 
# 5.Try to do data scaling for Kilometers driven
# 
# 6.Do the train test split
# 
# 7.Apply different ML regression Algorithms
# 
# 8.Calculate the error metrics.

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn as sk
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV


# In[6]:


car_data=pd.read_csv("Data_Train.csv")
car_test_data=pd.read_csv("Data_Test.csv")


# In[9]:


car_data


# In[10]:


car_data.describe()


# In[12]:


car_test_data


# In[13]:


car_test_data.describe()


# In[14]:


car_data.isnull().sum()


# In[15]:


car_test_data.isnull().sum()


# In[16]:


car_test_data[car_test_data['Power']=='null bhp']['Power'].count()


# In[17]:


car_data[car_data['Power']=='null bhp']['Power'].count()


# In[18]:


car_data['Power']=car_data['Power'].replace(to_replace="[null]" , value = np.NaN , regex = True)
car_test_data['Power']=car_test_data['Power'].replace(to_replace="[null]" , value = np.NaN , regex = True)
car_data.isnull().sum()


# In[19]:


car_test_data.isnull().sum()


# In[20]:


car_data['Mileage'].fillna(car_data['Mileage'].mode()[0] , inplace = True)
car_data['Engine'].fillna(car_data['Engine'].mode()[0] , inplace = True)
car_data['Power'].fillna(car_data['Power'].mode()[0] , inplace = True)
car_data['Seats'].fillna(car_data['Seats'].mode()[0] , inplace = True)
car_data.isnull().sum()


# In[21]:


car_test_data['Engine'].fillna(car_test_data['Engine'].mode()[0] , inplace = True)
car_test_data['Power'].fillna(car_test_data['Power'].mode()[0] , inplace = True)
car_test_data['Seats'].fillna(car_test_data['Seats'].mode()[0] , inplace = True)
car_test_data.isnull().sum()


# In[22]:


car_data['Mileage'] = car_data['Mileage'].str.split(" " , expand = True)
car_data['Mileage'] = car_data['Mileage'].astype("float")
car_data['Engine'] = car_data['Engine'].str.split(" " , expand = True)
car_data['Engine'] = car_data['Engine'].astype("float")
car_data['Power'] = car_data['Power'].str.split(" " , expand = True)
car_data['Power'] = car_data['Power'].astype("float")
car_data


# In[23]:


car_test_data['Mileage'] = car_test_data['Mileage'].str.split(" " , expand = True)
car_test_data['Mileage'] = car_test_data['Mileage'].astype("float")
car_test_data['Engine'] = car_test_data['Engine'].str.split(" " , expand = True)
car_test_data['Engine'] = car_test_data['Engine'].astype("float")
car_test_data['Power'] = car_test_data['Power'].str.split(" " , expand = True)
car_test_data['Power'] = car_test_data['Power'].astype("float")
car_test_data


# In[24]:


print("The total number of unqiue car :",str(len(car_data['Name'].unique())))


# In[25]:


car_data['Brand']=car_data['Name'].str.split(" ",2)
x=pd.DataFrame(car_data['Brand'])
car_data['Brand']=x['Brand'].str.get(0)
car_data['Model']=x['Brand'].str.get(1)
print("The car brands are: ",car_data['Brand'].unique())
print("The total number of unqiue car Brands :",str(len(car_data['Brand'].unique())))
print("The car brands are: ",car_data['Model'].unique())
print("The total number of unqiue car Brands :",str(len(car_data['Model'].unique())))


# In[26]:


car_data.Brand[car_data.Brand == 'Isuzu'] = 'ISUZU'
print("The total number of unqiue car Brands :",str(len(car_data['Brand'].unique())))


# In[27]:


training_set=car_data
test_set=car_test_data

car_data.info()


# Exploratory Data Analysis

# In[28]:


plt.rcParams["figure.figsize"]=[5,5]
plt.title("The distribution of prices of cars")
sb.boxplot(data=car_data['Price'],palette="pastel")


# In[30]:


plt.figure(figsize=(30,10))

c1=['orange','skyblue','yellow','lawngreen','violet','gold',]
c2=['hotpink','lime','red','sandybrown','grey']
plt.title("Brand Distribution")
sb.countplot(x=car_data['Brand'])


# In[31]:


sb.boxplot(x=car_data['Kilometers_Driven'])


# In[32]:


car_data['Kilometers_Driven'].astype('float64')
car_data['Kilometers_Driven']=car_data['Kilometers_Driven'].apply(lambda x: x if x <= 6000000 else -1)
car_data=car_data[car_data['Kilometers_Driven']!=-1]
car_data['Kilometers_Driven']=np.log(car_data['Kilometers_Driven'])
sb.boxplot(car_data['Kilometers_Driven'])


# In[33]:


c1=['orange','skyblue','yellow','lawngreen','violet','gold',]
c2=['hotpink','lime','red','sandybrown','grey']
plt.figure(figsize=(20,5))
plt.subplot(1,5,1)
glab=car_data['Location'].value_counts().index
gval=car_data['Location'].value_counts().values
plt.pie(gval,labels=glab,colors=c1,autopct='%1.1f%%')
plt.title("Location distribution")
plt.subplot(1,5,2)
glab=car_data['Owner_Type'].value_counts().index
gval=car_data['Owner_Type'].value_counts().values
plt.pie(gval,labels=glab,colors=c2,autopct='%1.1f%%')
plt.title("Owner_Type distribution")
plt.subplot(1,5,3)
glab=car_data['Fuel_Type'].value_counts().index
gval=car_data['Fuel_Type'].value_counts().values
plt.pie(gval,labels=glab,colors=c1,autopct='%1.1f%%')
plt.title("Fuel_Type distribution")
plt.subplot(1,5,4)
glab=car_data['Transmission'].value_counts().index
gval=car_data['Transmission'].value_counts().values
plt.pie(gval,labels=glab,colors=c2,autopct='%1.1f%%')
plt.title("Transmission distribution")
plt.subplot(1,5,5)
glab=car_data['Seats'].value_counts().index
gval=car_data['Seats'].value_counts().values
plt.pie(gval,labels=glab,colors=c2,autopct='%1.1f%%')
plt.title("Seats distribution")


# In[34]:


plt.figure(figsize=(30,20))

plt.subplot(1,2,1)
plt.title("Brand Distribution with Location")
sb.stripplot(y=car_data['Brand'],x=car_data['Price'],hue=car_data['Location'])
plt.subplot(1,2,2)
plt.title("Brand Distribution with Location")
sb.barplot(y=car_data['Brand'],x=car_data['Price'],hue=car_data['Transmission'])


# In[35]:


plt.figure(figsize=(25,10))
plt.subplot(1,2,1)
plt.title("Brand Distribution with Owner_Type")
sb.stripplot(y=car_data['Brand'],x=car_data['Price'],hue=car_data['Owner_Type'])
plt.subplot(1,2,2)
plt.title("Brand Distribution with Fuel_Type")
sb.stripplot(y=car_data['Brand'],x=car_data['Price'],hue=car_data['Fuel_Type'])


# In[36]:


plt.figure(figsize=(20,10))
plt.subplot(3,2,1)
plt.title("Price vs Location")
sb.boxplot(y=car_data['Location'],x=car_data['Price'])
plt.figure(figsize=(15,10))
plt.subplot(3,2,3)
plt.title("Prive vs location with influence of fuel_type")
sb.pointplot(y=car_data['Location'],x=car_data['Price'],hue=car_data["Fuel_Type"])
plt.subplot(3,2,4)
plt.title("Prive vs location with influence of Transmission_type")
sb.pointplot(y=car_data['Location'],x=car_data['Price'],hue=car_data["Transmission"])


# In[37]:


plt.figure(figsize=(10,10))
sb.swarmplot(y=car_data['Location'],x=car_data['Price'],hue=car_data["Owner_Type"])
plt.title("Price vs location with influence of Owner_type")


# In[38]:


g = sb.FacetGrid(car_data, col="Fuel_Type", row="Transmission")
g = (g.map(plt.scatter, "Price", "Seats", color="g").set_axis_labels("Total bill (in Lakhs)", "Seats"))


# In[39]:


Owners = {'First': 1,'Second': 2,'Fourth & Above': 4, 'Third': 3}
car_data['Owners'] = [Owners[item] for item in car_data['Owner_Type']]
car_data


# In[40]:


g = sb.FacetGrid(car_data, col="Fuel_Type", row="Transmission")
g = (g.map(plt.scatter, "Price", "Owners", color="b").set_axis_labels("Total bill (in Lakhs)", "Owners"))


# In[41]:


plt.figure(figsize=(15,7))
plt.subplot(1,4,1)
sb.regplot(x=car_data['Mileage'],y=car_data['Price'])
plt.title("Price vs Mileage")
plt.subplot(1,4,2)
sb.scatterplot(x=car_data['Mileage'],y=car_data['Price'],hue=car_data['Owner_Type'])
plt.subplot(1,4,3)
sb.scatterplot(x=car_data['Mileage'],y=car_data['Price'],hue=car_data['Fuel_Type'])
plt.subplot(1,4,4)
sb.scatterplot(x=car_data['Mileage'],y=car_data['Price'],hue=car_data['Transmission'])


# In[42]:


plt.figure(figsize=(15,7))
plt.subplot(1,4,1)
sb.regplot(x=car_data['Power'],y=car_data['Price'])
plt.title("Price vs Power")
plt.subplot(1,4,2)
sb.scatterplot(x=car_data['Power'],y=car_data['Price'],hue=car_data['Owner_Type'])
plt.subplot(1,4,3)
sb.scatterplot(x=car_data['Power'],y=car_data['Price'],hue=car_data['Fuel_Type'])
plt.subplot(1,4,4)
sb.scatterplot(x=car_data['Power'],y=car_data['Price'],hue=car_data['Transmission'])


# In[43]:


plt.figure(figsize=(15,7))
plt.subplot(1,4,1)
sb.regplot(x=car_data['Engine'],y=car_data['Price'])
plt.title("Price vs Engine")
plt.subplot(1,4,2)
sb.scatterplot(x=car_data['Engine'],y=car_data['Price'],hue=car_data['Owner_Type'])
plt.subplot(1,4,3)
sb.scatterplot(x=car_data['Engine'],y=car_data['Price'],hue=car_data['Fuel_Type'])
plt.subplot(1,4,4)
sb.scatterplot(x=car_data['Engine'],y=car_data['Price'],hue=car_data['Transmission'])


# Feature Selection:

# Extracting important features in our dataset :

# In[45]:


car_data_X=car_data.loc[:, car_data.columns.isin(['Location', 'Year', 'Kilometers_Driven', 'Fuel_Type','Transmission', 'Owner_Type', 'Brand','Model'])]
car_data_y=car_data['Price']
car_data_X


# In[46]:


car_data_y


# In[47]:


one_hot = pd.get_dummies(car_data_X['Location'],prefix='Location')
car_data_X = car_data_X.drop('Location',axis = 1)
car_data_X=car_data_X.join(one_hot)
two_hot = pd.get_dummies(car_data_X['Fuel_Type'],prefix='Fuel')
car_data_X = car_data_X.drop('Fuel_Type',axis = 1)
car_data_X=car_data_X.join(two_hot)
three_hot = pd.get_dummies(car_data_X['Owner_Type'],prefix='Owner')
car_data_X = car_data_X.drop('Owner_Type',axis = 1)
car_data_X=car_data_X.join(three_hot)
two_hot = pd.get_dummies(car_data_X['Transmission'])
car_data_X = car_data_X.drop('Transmission',axis = 1)
car_data_X=car_data_X.join(two_hot)


# In[48]:


car_data_X


# In[49]:


from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
car_data_X["Brand"] = lb_make.fit_transform(car_data_X["Brand"])
car_data_X["Model"] = lb_make.fit_transform(car_data_X["Model"])


# In[50]:


car_data_X


# In[51]:


car_data_X['Year']=car_data_X['Year'].apply(lambda x: 2019-x)
car_data_X


# In[52]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X=car_data_X.astype('int')
y=car_data_y.astype('int')


# In[53]:


bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']
featureScores


# In[54]:


print(featureScores.nlargest(25,'Score'))


# In[55]:


from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X,y)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(25).plot(kind='barh')
plt.show()


# In[56]:


corrmat = car_data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10,10))
g=sb.heatmap(car_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[65]:


car_test_data.columns


# In[67]:


car_test_data['Brand']=car_test_data['Name'].str.split(" ",2)
x=pd.DataFrame(car_test_data['Brand'])
car_test_data['Brand']=x['Brand'].str.get(0)
car_test_data['Model']=x['Brand'].str.get(1)


car_test_data["Brand"] = lb_make.fit_transform(car_test_data["Brand"])
car_test_data["Model"] = lb_make.fit_transform(car_test_data["Model"])


# In[68]:


car_test_data=car_test_data.drop(columns=['Name'])
car_test_data['Kilometers_Driven']=np.log(car_test_data['Kilometers_Driven'])


# In[69]:


car_test_data['Year']=car_test_data['Year'].apply(lambda x: 2019-x)
car_test_data


# Data Splitting:

# In[70]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(car_data_X, car_data_y, test_size = 0.2)

X_train.info()


# Model 1: Linear Regression

# In[71]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
lreg = LinearRegression()
lreg.fit(X_train,y_train)
y_pred_linear = lreg.predict(X_test)
print('The Score on the train set with a linear regression is:', lreg.score(X_train,y_train))
print('The Score on the test set with a linear regression is:', lreg.score(X_test,y_test))
print("Accuracy :",lreg.score(X_test,y_test)*100,'%')
print("Mean squared error (rmse): %.2f" % np.sqrt(mean_squared_error(y_test, y_pred_linear)))
print("The r2_score is: ", r2_score(y_test, y_pred_linear))


# Model 2: Random Forest Regressor

# In[73]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train,y_train)
y_pred_rf=regressor.predict(X_test)
print('The Score on the training set with  a Random Forest  regression is:', regressor.score(X_train,y_train))
print(' The Score on the test set with a hyperparameter optimized random forest regressor is:',regressor.score(X_test,y_test))
print("Accuracy :",regressor.score(X_test,y_test)*100,'%')
print("Mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("The r2_score is: ", r2_score(y_test, y_pred_rf))


# In[74]:


from sklearn.model_selection import RandomizedSearchCV
RFR = RandomForestRegressor()

n_estimators = [1 , 2 , 4, 8 , 16, 32, 64, 100, 200]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rf_random = RandomizedSearchCV(estimator = RFR, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

best_model = rf_random.fit(X_train, y_train)


# In[75]:


best_model.best_params_


# In[76]:


rand_est = RandomForestRegressor(n_estimators= 100, min_samples_split= 5, min_samples_leaf= 1, max_features= 'auto',max_depth=100,bootstrap=True)
rand_est.fit(X_train,y_train)
y_pred_rfr = rand_est.predict(X_test)
print(' The Score on the train set with a hyperparameter optimized random forest regressor is:',rand_est.score(X_train,y_train))
print(' The Score on the test set with a hyperparameter optimized random forest regressor is:',rand_est.score(X_test,y_test))
print("Mean squared error: %.2f"% np.sqrt(mean_squared_error(y_test, y_pred_rfr)))


# Model 4: K-Neighbours

# In[78]:


neigh = KNeighborsRegressor(n_neighbors=15)
neigh.fit(X_train,y_train)
k_pred=neigh.predict(X_test)
print('The Score on the test set with a Kneighbours  regression is:', neigh.score(X_test,y_test))
print("Accuracy :",neigh.score(X_test,y_test)*100,'%')
# The Root mean squared error
print("Mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test,k_pred )))
print("The r2_score is: ", r2_score(y_test, k_pred))


# Model 5: Lasso Regression

# In[79]:


from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.15)
clf.fit(X_train,y_train)
lasso_predict=clf.predict(X_test)
print('The Score on the test set with a Lasso  regression is:', clf.score(X_test,y_test))
print("Accuracy :",clf.score(X_test,y_test)*100,'%')
# The Root mean squared error
print("Mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, lasso_predict)))
print("The r2_score is: ", r2_score(y_test, lasso_predict))


# Model6: SVM

# In[80]:


from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
print('The Score on the test set with a Support Vector regression is:', regressor.score(X_test,y_test))
print("Accuracy :",regressor.score(X_test,y_test)*100,'%')
# The Root mean squared error
print("Mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
print("The r2_score is: ", r2_score(y_test, y_pred))


# In[81]:


app1_test=car_test_data
app1_test


# In[82]:


app1_test['Fuel_Electric']=0
app1_test=app1_test[car_data_X.columns]
app1_test.columns
app12_test=app1_test
app12_test


# In[89]:


car_data_X.columns

