import csv
import numpy as np
from numpy import *
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt
import math
import seaborn as sns
#from scipy import stats
import scipy.stats as stats
from numpy import linalg as LA
import statsmodels.api as sm
sns.set(color_codes=True)


import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc

arr = np.loadtxt(open("winequality-white.csv", "rb"), delimiter=",", skiprows=1)
b = arr[:,:11]
temp = np.ones((4898,1))
idV = b
idv = np.concatenate((temp,b),axis = 1)
dv = arr[:,11:]

variables=["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide" , "total sulfur dioxide","density", "pH" , "sulphates", "alcohol"]
#print idv, idv.shape
#print dv, dv.shape
'''
for i in range(1,12):
	sns.distplot(idv[:,i],kde=True)
	plt.title("Distribution of "+variables[i-1])
	plt.savefig('/home/sumanth/Documents/SDS/'+variables[i-1]+' Distribution'+'.png')
	plt.show()

'''
sns.distplot(dv,kde=False)
plt.title("Distribution of Quality ")
plt.savefig('/home/sumanth/Documents/SDS/'+'Quality Distribution'+'.png')
plt.show()
#Normalize'''
print idv
'''for i in range(1,12):
	k=np.mean(idv[:,i])
	l=np.var(idv[:,i])
	idv[:,i]=np.divide(np.subtract(idv[:,i],k),np.sqrt(l))

print idv'''


idv_transpose =  idv.transpose()

x_t_x = np.dot(idv_transpose,idv)

x_t_x_inv = inv(x_t_x)
#print x_t_x_inv

temp1 = np.dot(x_t_x_inv,idv_transpose)

beta_coeff = np.dot(temp1,dv)

print"Regression Coefficients"
print beta_coeff
t=np.zeros((12,1))
for i in range(12):
	t[i]=i
#sns.distplot(beta_coeff)
#plt.show()

y_cap = np.dot(idv,beta_coeff)

print"Estimated Y Value"
print y_cap



residual = np.subtract(dv, y_cap)

print"Residual Errors"
print residual

import pylab


#sns.distplot(residual)
#plt.title("Distribution of Residuals")
#plt.savefig('/home/sumanth/Documents/SDS/'+'dist_residual'+'.png')
#plt.show()
'''plt.scatter(y_cap,residual)
plt.title("Y_hat vs Errors")
plt.savefig('/home/sumanth/Documents/SDS/'+'Y_hat vs Errors'+'.png')
plt.show()'''
#Confidence Interval for Regression Coefficients

CI = 0.95
alpha = 1-CI

n, p = idv.shape
print"shapes:", n,p

cjj = np.diag(x_t_x_inv)
print cjj

SSE = np.dot(residual.transpose(),residual)

print SSE
se_2 = SSE / (n-p-1)

print "S_E2 :",se_2

low_reg=[]
high_reg=[]

t_alp = 2.262
for i in range(p):
	low_reg.append( beta_coeff[i] - (t_alp * ((se_2 * cjj[i]) ** 0.5)) )
	high_reg.append( beta_coeff[i] + (t_alp * ((se_2 * cjj[i]) ** 0.5)) )
	#print "beta_",i, low_reg, high_reg

print low_reg, high_reg

#Model Adequacy Test

y_mean = np.mean(dv)

print y_mean

#SST = 0

#for i in range(n):
#	SST = SST + ((dv[i]-y_mean) ** 2)


y_sub_Ybar = np.subtract(dv,y_mean)


SST = np.dot(y_sub_Ybar.transpose(),y_sub_Ybar)

print y_sub_Ybar

print SST

SSR = np.subtract(SST, SSE)

R_2 = np.divide(SSR,SST)

print "R^2", R_2

adj_r2 = 1 - ((SSE/(n-p-1))/(SST/(n-1)))

print adj_r2

idv_mean=[]
idv_var=[]
for i in range(p):
	idv_mean.append(np.mean(idv[:,i]))
	idv_var.append(np.var(idv[:,i]))

print "idv",idv_mean , idv_var

y_values = range(12)

variables=["constant","fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide" , "total sulfur dioxide","density", "pH" , "sulphates", "alcohol"]


'''for i in range(p):
	plt.title('Y_vs_'+variables[i])
	plt.scatter(idv[:,i:i+1],dv)
	plt.savefig('/home/sumanth/Documents/SDS/'+'Y_vs_'+variables[i]+'.png')
	plt.show()'''


"""bins = np.linspace(0, 500)

plt.hist(list(dv), bins, alpha=0.5, label='y')
plt.hist(list(y_cap), bins, alpha=0.5, label='y_cap')
plt.legend(loc='upper right')
plt.show()"""
X=[]
variables=["constant","fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide" , "total sulfur dioxide","density", "pH" , "sulphates", "alcohol"]

for i in range(1,4899):
	X.append(i)

"""for i in range(p):
	plt.scatter(X,list(idv[:,i:i+1]))
	plt.title(variables[i])
	plt.legend(["mean "+str("%.4f" % idv_mean[i])+" variance "+str("%.4f" % idv_var[i])])
	
	
	plt.savefig('/home/sumanth/Documents/SDS/'+variables[i]+'.png')
	plt.show()"""



#plt.scatter(X,residual)
#plt.scatter(X,list(dv), color='g')
#plt.scatter(X, list(y_cap), color='orange')
#plt.savefig('/home/sumanth/Documents/SDS/'+'Y_vs_Ycap'+'.png')
#plt.show()

#plot_mean_and_CI(beta_coeff[0],high_reg[0],low_reg[0],color_mean='b',color_shading='b')

#pltshow()
beta_var = np.multiply(se_2,x_t_x_inv)
#print beta_var
#dist=pd.Dataframe([low_reg[0],high_reg[0],beta_coeff[0]],beta)


#R=np.dot(idv.transpose(),idv)
#R=np.divide(R,n-1)
#print R

idV_mean=[]
idV_var=[]

for i in range(p-1):
	idV_mean.append(np.mean(idV[:,i]))
	idV_var.append(np.var(idV[:,i]))


X_star = np.subtract(idV,idV_mean)
#print X_star.shape

S = np.dot(X_star.transpose(),X_star)
S = np.true_divide(S, n-1)

#print S.shape, S

R = np.zeros((p-1,p-1))
for i in range(p-1):
	for j in range(p-1):
		R[i][j] = S[i][j]/math.sqrt(S[i][i]*S[j][j])

print R


'''for i in range(n):
	for j in range(1,p):
		X_[i][j]=float(idv[i][j]-idv_mean[j])/float(math.sqrt(idv_var[j]))
X_ = X_[:,1:]

#print X_

R_=np.dot(X_.transpose(),X_)
R_=np.divide(R_,n-1)

print R_ '''

U, sigma, VH = linalg.svd(R, full_matrices=False)
print "sigma",sigma

'''plt.scatter(range(1,12),sigma,c='red',s=44)
plt.title('Eigen values plot')
plt.savefig('/home/sumanth/Documents/SDS/eigen_values'+'.png')
plt.show()
plt.scatter(range(1,12),sigma,c='green',s=44)
plt.plot(range(1,12),sigma)
plt.title('Eigen values plot')
plt.savefig('/home/sumanth/Documents/SDS/eigen_values_plot'+'.png')
plt.show()'''
#print np.corrcoef(idv).shape


MCN = sigma[0]/sigma[-1]

print MCN


influ_point = np.dot(idv,x_t_x_inv)  ## influ_point = H
influ_point = np.dot(influ_point,idv.transpose())
dia_influ = diag(influ_point)
print np.sum(dia_influ)
count=0
for i in dia_influ:
	if i> 2*(p+1)/n:
		count+=1
	else:
		print i

print "count" , count

obs = range(1,4899)
#plt.scatter(obs, dia_influ)
#plt.title('Influential Points')
#plt.savefig('/home/sumanth/Documents/SDS/Influential_Points'+'.png')
#plt.show()
#####cooks distance
k=np.zeros((n,1))
for i in range(n):
	k[i]=influ_point[i][i]
print k.shape
print residual.shape
R=np.subtract(1,k)
R=np.multiply(se_2,R)
R=np.sqrt(R)
R=np.divide(residual,R)
print "R" , R ,"R"

D=np.zeros((n,1))
for i in range(n):
	D[i]=(R[i]/p)*(k[i]*(1-k[i]))

print D
'''plt.plot(range(1,n+1),D)
plt.title("Cook's Distance")
plt.savefig('/home/sumanth/Documents/SDS/cooksdistance'+'.png')
plt.show()'''

count=0
for i in D:
	if i<1:
		count+=1

print count	

#################################

row=["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide" , "total sulfur dioxide","density", "pH" , "sulphates", "alcohol"]

col=["Mean","Median","Min","Max","stdev"]

cellvalues=[]
for i in range(1,len(row)+1):
	l=[]
	s=idv[:,i]
	l.append(np.mean(s))
	l.append(np.median(s))
	l.append(np.amin(s))
	l.append(np.amax(s))
	l.append(np.std(s))
	cellvalues.append(l)


#print cellvalues



#plt.table(cellText=cellvalues,rowLabels=row,colLabels=col)
#plt.show()
#print diag(influ_point), influ_point.shape 

from sklearn.cluster import KMeans  
'''
plt.scatter(,dv, label='True Position')
plt.show()
kmeans = KMeans(n_clusters=3)  
kmeans.fit(dv)  
print(kmeans.cluster_centers_)  
plt.scatter(obs,dv, c=kmeans.labels_, cmap='rainbow')  
plt.show()
plt.scatter(kmeans.cluster_centers_,color='black')
plt.show()

import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))  
plt.title("Customer Dendograms")
dend = shc.dendrogram(shc.linkage(b, method='ward'))  
plt.show()  '''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

from sklearn import cluster
top_K=["volatile acidity","chlorides"]

#print b

for i in range(1,11):
	j=np.zeros((n,1))
	j = b[:,i:i+1]
	j= np.concatenate((j,dv),axis=1)
	#j=np.concatenate((j,dv),axis=1)

	k_means = cluster.KMeans(n_clusters=3)
	k_means.fit(j)
	#b["volatile acidity"] 
	label = k_means.labels_
	print(k_means.cluster_centers_)
	C = k_means.cluster_centers_
	#plt.scatter(dv,k_means.labels_, c=k_means.labels_,color='')
	#fig = plt.figure()
	#ax = Axes3D(fig)
	colors = ['red' if i == 0 else 'yellow' if i==1 else 'black'for i in label]	
	plt.scatter(j[:,0],j[:,1],c=colors)
	#\ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)
	plt.show()
