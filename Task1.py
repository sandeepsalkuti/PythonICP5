#importing required modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("train.csv")                #reading dataset
garage = data["GarageArea"]                    #taking x value
saleprice = data["SalePrice"]                  #taking y value
print("Before outliers removing")
plt.scatter(garage, saleprice, color="blue")   #plotting scatter plot against x and y
plt.xlabel('Garage Area')                      #labelling x axis
plt.ylabel('Sale Price')                       #labelling y axis
plt.title('Linear Regression Model')
plt.show()                                     #showing plot
lowerbound = 0.1                               #declaring lower bound value
upperbound = 0.95                              #declaring upper bound value
outlier = garage.quantile([lowerbound, upperbound]) #applying lower and upperbounds on garage column using quantile
final = ((garage>outlier.loc[lowerbound]) & (garage<outlier.loc[upperbound])) #filtering location of values those satify the condition of lower&upper bounds
result = garage[final]                         #fetching those values
print("After outliers removing")
plt.scatter(garage[final], saleprice[final])   #plotting fetched values against saleprice column
plt.xlabel('Garage Area')                       #labelling x axis
plt.ylabel('Sale Price')                        #labelling y axis
plt.title('Linear Regression Model')
plt.show()