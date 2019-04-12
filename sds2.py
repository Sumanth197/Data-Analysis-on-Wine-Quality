import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error


data=pd.read_csv("winequality-white.csv")
#sns.pairplot(data)
#plt.savefig('/home/sumanth/Documents/SDS/'+'pairwiseplot'+'.png')
#plt.show()
feature_names=[]
feature_names = data.columns[:-1]
#print data.iloc[:,:-1] 
#print feature_names
df = pd.DataFrame(data.iloc[:,:-1], columns=feature_names)

#print df.isnull().sum()
#print df.describe()
target = pd.DataFrame(data.quality, columns=["quality"])

df_all = pd.concat([df,target],axis=1)
#print df_all

X = df[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide" , "total sulfur dioxide","density", "pH" , "sulphates", "alcohol"]]
y = target["quality"]
X = sm.add_constant(X)


model = sm.OLS(y, X).fit()
predictions = model.predict(X)

print model.summary()

#PCA Block
X_std = StandardScaler().fit_transform(X)
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)

#cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
#print('Eigenvalues in descending order:')
#for i in eig_pairs:
#    print(i)

#print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))
top_eigen_values=["volatile acidity","chlorides","density", "pH" , "sulphates", "free sulfur dioxide" , "total sulfur dioxide"]

df1 = pd.DataFrame(data.iloc[:,:], columns=top_eigen_values)
df1 = sm.add_constant(df1)

#print df1

model1 = sm.OLS(y, df1).fit()
predictions = model1.predict(df1)

print model1.summary()

#END OF PCA Block


#VIF Block
enp=["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide" , "total sulfur dioxide","density", "pH" , "sulphates", "alcohol"]
for i in enp:
	tar=pd.DataFrame(X[i], columns=[i])
	df_temp = X
	df_temp=df_temp.drop(columns=[i])
	temp_model = sm.OLS(tar, df_temp).fit()
	predictions = temp_model.predict(df_temp)
	#print temp_model.summary()	
	


Rj_2={"fixed acidity":0.628, "volatile acidity":0.124, "citric acid":0.142,"residual sugar":0.921, "chlorides":0.191,"free sulfur dioxide":0.441,"total sulfur dioxide":0.553,"density":0.965,"pH":0.545,"sulphates":0.122,"alcohol":0.870}

vif=[]

for i in Rj_2.keys():
	VIF = 1/(1-Rj_2[i])
	vif.append(VIF)
	print "VIF %s"%i, VIF
	
plt.scatter(range(1,12),vif,c='green',s=44)
plt.plot(range(1,12),vif)
plt.title("Variance Inflation Factor")
plt.savefig('/home/sumanth/Documents/SDS/'+'VIF_each_variable'+'.png')
plt.show()

drop_col = ["density","residual sugar"]

X_aft = X.drop(columns=drop_col)
#print X_aft

model3 = sm.OLS(y, X_aft).fit()
predictions = model3.predict(X_aft)

print model3.summary()

#END OF VIF Block


drop_col_p = ["citric acid", "chlorides", "total sulfur dioxide"]

X_aft1 = X.drop(columns=drop_col_p)
#print X_aft

model4 = sm.OLS(y, X_aft1).fit()
predictions = model4.predict(X_aft1)

print model4.summary()
	
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="Purples")
plt.title("correlation between variables")

plt.savefig('/home/sumanth/Documents/SDS/'+'corr'+'.png')
plt.show()


########### AI Block ########### 80-20 Splitting 

from sklearn.model_selection import train_test_split
train, test = train_test_split(df_all, test_size=0.2, random_state=4) 

y = train["quality"]
cols = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide" , "total sulfur dioxide","density", "pH" , "sulphates", "alcohol"]
X = train[cols]

regr = linear_model.LinearRegression()
regr.fit(X,y)

ytrain_pred = regr.predict(X)
print("In-sample Mean squared error: %.2f"
      % mean_squared_error(y, ytrain_pred))


ytest = test["quality"]
cols = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide" , "total sulfur dioxide","density", "pH" , "sulphates", "alcohol"]

Xtest=test[cols]

ypred = regr.predict(Xtest)
print("Out-of-sample Mean squared error: %.2f"
      % mean_squared_error(ytest, ypred))


########### AI Block ########### 60-40 Splitting 

from sklearn.model_selection import train_test_split
train, test = train_test_split(df_all, test_size=0.4, random_state=4) 

y = train["quality"]
cols = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide" , "total sulfur dioxide","density", "pH" , "sulphates", "alcohol"]
X = train[cols]

regr = linear_model.LinearRegression()
regr.fit(X,y)

ytrain_pred = regr.predict(X)
print("In-sample Mean squared error: %.2f"
      % mean_squared_error(y, ytrain_pred))


ytest = test["quality"]
cols = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide" , "total sulfur dioxide","density", "pH" , "sulphates", "alcohol"]

Xtest=test[cols]

ypred = regr.predict(Xtest)
print("Out-of-sample Mean squared error: %.2f"
      % mean_squared_error(ytest, ypred))

from mpl_toolkits.mplot3d import Axes3D 


from sklearn import cluster
top_K=["volatile acidity","chlorides","quality"]

b = pd.DataFrame(data.iloc[:,:], columns=top_K)
#print b
k_means = cluster.KMeans(n_clusters=3)
k_means.fit(b)
#b["volatile acidity"] 
print(k_means.labels_)
print(k_means.cluster_centers_)

plt.scatter(target,k_means.labels_, c=k_means.labels_,color='')
#ax.scatter(b[:,0],b[:,1],b[:,2])
plt.scatter(k_means.cluster_centers_,color='black')

plt.show()


df.describe().to_csv("my_description.csv")




############## Logistic Regression ############

'''

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X, y)
print clf.score(X, y)


from sklearn.linear_model import LogisticRegressionCV

clf = LogisticRegressionCV(cv=5, random_state=0, multi_class='multinomial').fit(X, y)
print clf.score(X, y) 

###################################################3



from sklearn.tree import DecisionTreeRegressor, export_graphviz

X = df[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide" , "total sulfur dioxide","density", "pH" , "sulphates", "alcohol"]]
y = target["quality"]
dt = DecisionTreeRegressor(min_samples_split=6, random_state=99)
dt.fit(X, y)


X_test = np.random.rand(3,11)	
print X_test
y_1 = dt.predict(X_test)

print y_1

# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
#plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
#print dt.predict(X)	
'''





