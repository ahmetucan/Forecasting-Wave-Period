# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 17:54:47 2021

@author: Ahmet
"""

#1.kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


# Veri Yükleme
veriler = pd.read_excel('veriler.xlsx')

s = veriler.iloc[:,1:2]
Y = veriler.iloc[:,3:4].values
y= veriler.iloc[:,3:4]
z = veriler.iloc[:,7:8]
x = pd.concat([s,z], axis =1)
X = x.values




# Linear Regression doğrusal model oluşturma 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)
y_pred = lin_reg.predict(X)

plt.scatter(X[:,:1],Y, color = 'red')
plt.scatter(X[:,:1], y_pred, color = 'blue')
plt.title('(LR)Hs-T Tahmin Grafiği')
plt.xlabel('Periyot (s)')
plt.ylabel('Belirgin Dalga Yüksekliğ(Hs)(m)')
plt.legend(['Ölçüm','Tahmin'])
plt.show()

print('Linear Regression R2 Değeri')
print(r2_score(Y,lin_reg.predict(X)))

# Polinomal Regression- doğrusal 2. dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree= 2 )
x_poly = poly_reg.fit_transform(X)
#print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
y_pred = lin_reg2.predict(x_poly)
plt.scatter(X[:,:1],Y, color = 'red')
plt.scatter(X[:,:1], y_pred, color = 'blue')
plt.title('(PR=2)Hs-T Tahmin Grafiği')
plt.xlabel('Periyot (s)')
plt.ylabel('Belirgin Dalga Yüksekliğ(Hs)(m)')
plt.legend(['Ölçüm','Tahmin'])
plt.show()

print('2 degree Polinomal regression R2 Değeri')
print(r2_score(y,lin_reg2.predict(x_poly)))

# 4. dereceden polinomal regresyon

poly_reg3 = PolynomialFeatures(degree = 4 )
x_poly3 = poly_reg3.fit_transform(X)
#print(x_poly)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)
y_pred = lin_reg3.predict(x_poly3)
plt.scatter(X[:,:1],Y, color = 'red')
plt.scatter(X[:,:1], y_pred, color = 'blue')
plt.title('(PR=4)Hs-T Tahmin Grafiği')
plt.xlabel('Periyot (s)')
plt.ylabel('Belirgin Dalga Yüksekliğ(Hs)(m)')
plt.legend(['Ölçüm','Tahmin'])
plt.show()

print('4 degree Polinomal regression R2 Değeri')
print(r2_score(y,lin_reg3.predict(x_poly3)))


# verilerin ölçeklendirilmesi
from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()
x_olcekli= sc1.fit_transform(X)
sc2 =StandardScaler()
y_olcekli = sc2.fit_transform(Y)


# Support Vector Regression
from sklearn.svm import SVR

svr_reg = SVR(kernel ='rbf')
svr_reg.fit(x_olcekli,y_olcekli)
y_pred = svr_reg.predict(x_olcekli)


plt.scatter(x_olcekli[:,:1],y_olcekli, color = 'red')
plt.scatter(x_olcekli[:,:1], y_pred, color = 'blue')
plt.title('(SVR)Hs-T Tahmin Grafiği')
plt.xlabel('periyot')
plt.ylabel('Belirgin Dalga Yüksekliğ(Hs_ölcekli)')
plt.legend(['Ölçüm','Tahmin'])
plt.show()



print('SVR R2 Değeri')
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))

# Decison Tree Regression
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
y_pred = r_dt.predict(X)
plt.scatter(X[:,:1],Y, color = 'red')
plt.scatter(X[:,:1], y_pred, color = 'blue')
plt.title('(DT)Hs-T Tahmin Grafiği')
plt.xlabel('Periyot (s)')
plt.ylabel('Belirgin Dalga Yüksekliğ(Hs)(m)')
plt.legend(['Ölçüm','Tahmin'])
plt.show()

print('Decision Tree R2 değeri')
print(r2_score(Y,r_dt.predict(X)))


# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rf_reg= RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X,Y.ravel())
y_pred = rf_reg.predict(X)
plt.scatter(X[:,:1],Y, color = 'red')
plt.scatter(X[:,:1], y_pred, color = 'blue')
plt.title('(RF)Hs-T Tahmin Grafiği')
plt.xlabel('Periyot (s)')
plt.ylabel('Belirgin Dalga Yüksekliğ(Hs)(m)')
plt.legend(['Ölçüm','Tahmin'])
plt.show()



print('Random Forest R2 Değeri')
print(r2_score(Y,rf_reg.predict(X)))



# Özet R2 Değerleri
print('---------------------')
print('Linear Regression R2 Değeri')
print(r2_score(Y,lin_reg.predict(X)))

print('2 degree Polinomal regression R2 Değeri')
print(r2_score(y,lin_reg2.predict(x_poly)))

print('4 degree Polinomal regression R2 Değeri')
print(r2_score(y,lin_reg3.predict(x_poly3)))

print('SVR R2 Değeri')
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))

print('Decision Tree R2 değeri')
print(r2_score(Y,r_dt.predict(X)))

print('Random Forest R2 Değeri')
print(r2_score(Y,rf_reg.predict(X)))
