import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from pandas import Series, DataFrame
data_train = pd.read_csv("C:/Users/310243978/Documents/R/Machine Learning/Kaggle/Titanic/train.csv")
print(data_train)
data_train.info()
data_train.describe()

fig = plt.figure()
fig.set(alpha = 0.2)

plt.subplot2grid((2,3),(0,0))
data_train.Survived.value_counts().plot(kind = 'bar')
plt.title(u"survival,1=survived")
plt.ylabel(u'count')

plt.subplot2grid((2,3),(0,1))# 2x3 layout
data_train.Pclass.value_counts().plot(kind="bar")
plt.title(u'Passenger Class')
plt.ylabel(u'Count')
'''
plt.subplot2grid((2,2),(1,0))
data_train.Sex.value_counts().plot(kind = 'bar')
plt.title(u'Sex')
plt.ylabel(u'Count')
'''

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"Age")                         # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y')
plt.title(u"Survival by Age (1=Survived)")

plt.subplot2grid((2,3),(1,0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind = 'kde')
data_train.Age[data_train.Pclass == 2].plot(kind = 'kde')
data_train.Age[data_train.Pclass == 3].plot(kind = 'kde')
plt.xlabel(u"Age")# plots an axis lable
plt.ylabel(u"density")
plt.title(u"density by Age")
plt.legend((u'1st class', u'2nd class',u'3rd class'),loc='best') # sets our legend for our graph.

plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"People by embark")
plt.ylabel(u"Count")
plt.show()

#check the survival by Passenger Class
fig = plt.figure()
fig.set(alpha=0.2)

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'Survived':Survived_1, u'NonSurvived':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"Survival by PClass")
plt.xlabel(u"P Class")
plt.ylabel(u"Count")
plt.show()

#check the survival by Sex
fig = plt.figure()
fig.set(alpha=0.2)

Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
df=pd.DataFrame({u'Female':Survived_f, u'Male':Survived_m})
df.plot(kind='bar', stacked=True)
plt.title(u"Survival by Sex")
plt.xlabel(u"Sex")
plt.ylabel(u"Count")
plt.show()

#by sex by PClass by Sex
fig=plt.figure()
fig.set(alpha=0.2) #set the transparance
plt.title(u"Survival by PClass by Sex")
ax1=fig.add_subplot(141)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female, highclass", color='#FA2479')
ax1.set_xticklabels([u"Survived", u"Unsurvived"], rotation=0)
ax1.legend([u"Femail/HihgClass"], loc='best')
ax2=fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
ax2.set_xticklabels([u"Unsurvived", u"Survived"], rotation=0)
plt.legend([u"Female/LowClass"], loc='best')
ax3=fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
ax3.set_xticklabels([u"Unsurvived", u"Survived"], rotation=0)
plt.legend([u"Male/HighClass"], loc='best')
ax4=fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
ax4.set_xticklabels([u"Unsurvived", u"Survived"], rotation=0)
plt.legend([u"Male/LowClass"], loc='best')
plt.show()

#check By Embark
Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df2 = pd.DataFrame({u"NotSurvived":Survived_0, u"Survived":Survived_1})
print(df2)

#check by SibSp
'''
Survived_0 = data_train.SibSp[data_train.Survived == 0].value_counts()
Survived_1 = data_train.SibSp[data_train.Survived == 1].value_counts()
df3 = pd.DataFrame({u"NotSurvived":Survived_0, u"Survived":Survived_1})
print(df3)
'''
g = data_train.groupby(['SibSp','Survived'])
df3 = pd.DataFrame(g.count()['PassengerId'])
print(df3)
#check by Parch
Survived_0 = data_train.Parch[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Parch[data_train.Survived == 1].value_counts()
df4 = pd.DataFrame({u"NotSurvived":Survived_0, u"Survived":Survived_1})
print(df4)

#check by Cabin
fig = plt.figure()
fig.set(alpha = 0.2)

Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_Nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df = pd.DataFrame({u'HasCabin':Survived_cabin,u'NoCabin':Survived_Nocabin}).transpose() #转置

df.plot(kind='bar', stacked=True)
plt.title(u"Survival by Cabin")
plt.xlabel(u"Cabin has or not")
plt.ylabel(u"Count")
plt.legend((u"No",u"Yes"),loc='best')
plt.show()


