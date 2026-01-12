import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

#shape of data
df = pd.read_csv("cep1_dataset.csv")
print(df.shape)

#Finding missing values
missing_val = df.isnull()
print(missing_val)

#Remove duplicates
duplicates = df[df.duplicated()]
print(duplicates)
df = df.drop_duplicates()
print(df.shape)

#Central tendency plots
## Mean plot
mean_data = df.mean()
plt.plot(mean_data)
plt.ylabel("Mean")
plt.show()
## Mode Plot
mode_data = df.mode()
plt.plot(mode_data)
plt.ylabel("Mode")
plt.show()
## Meadian plot
median_data = df.median()
plt.plot(median_data)
plt.ylabel("Meadian")
plt.show()

#Seperating catogerical data

categorical_data = df.select_dtypes(exclude=[np.number])
print(categorical_data.shape)

#scatter plot for occurrence of CVD across the Age category
plt.scatter(df['age'],df['target'])
plt.xlabel('age')
plt.ylabel('occurrence of CVD')
plt.show()

#composition of all patients with respect to the Sex category
sex_catogery = df['sex'].value_counts()
print("Number of males,Females are :",sex_catogery[1],",",sex_catogery[0]) 

#scatter plot for heart attacks based on anomalies in the resting blood pressure of a patient
plt.scatter(df['trestbps'],df['target'])
plt.xlabel('resting blood pressure of a patient')
plt.ylabel('occurrence of CVD')
plt.show()

#Scatter plot to show the relationship exists between Cholestrol and the occurrence of a heart attack
plt.scatter(df['chol'],df['target'])
plt.xlabel('cholestrol')
plt.ylabel('occurrence of CVD')
plt.show()

#relationship exists between peak exercising and the occurrence of a heart attack
plt.scatter(df['slope'],df['target'])
plt.xlabel('peak exercising')
plt.ylabel('occurrence of CVD')
plt.show()

#To check if thalassemia is a major cause of CVD  
plt.scatter(df['thalach'],df['target'])
plt.xlabel('thalassemia')
plt.ylabel('occurrence of CVD')
plt.show()

#pairplot for all variables
sns.pairplot(df,hue='target')
plt.show()

#Prediction for the risk of heart attack

#Correlation  based feature selection
corr = df.corr()
cor_matrix = df.corr().abs()

#Upper traingular matrix
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))

# significant variables selection with high correlation
Significant_variables = [column for column in upper_tri.columns if any(upper_tri[column] < 0.1)]
print(Significant_variables)

#Creating dataset with significant variable selection 
X = df[['sex','cp','trestbps','chol','fbs','restecg','thalach','exang', 'oldpeak','slope','ca','thal']]
Y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.28,random_state=1)
#Predictions using Logistic Regression
model = sm.Logit(endog=y_train, exog=X_train).fit()

print(model.summary())

pred = model.predict(exog=X_test)
print(pred.head())

print("y_test Vs Predicted labels for Logistic Regression:")
print(list(y_test))
print(list(round(pred)))

conf_mat = confusion_matrix(y_true=list(y_test), y_pred=list(round(pred)))
print("Confusion Matrix for Logistic Regression :")
print(conf_mat)

print("Logistic Regression Accuracy Score: ",accuracy_score(y_true=list(y_test), y_pred=list(round(pred))))

#Predictions using Random Forest Classifier
clf=RandomForestClassifier(n_estimators=170)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("y_test Vs Predicted labels for RandomForestClassifier:")
print(list(y_test))
print(list(y_pred))
conf_mat = confusion_matrix(y_true=list(y_test), y_pred=list(y_pred))
print("Confusion Matrix for RandomForestClassifier :")
print(conf_mat)

print("Random Forest Accuracy: ",accuracy_score(y_true=list(y_test), y_pred=list(y_pred)))
