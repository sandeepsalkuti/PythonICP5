
import pandas as pd
import numpy as np
import math
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score                  #importing all necessary modules for performance evaluation

train = pd.read_csv('winequality-red.csv')                               #reading dataset

# Handling missing value
data = train.select_dtypes(include=[np.number]).interpolate().dropna()   #dropping null values

# Top 3 correlation
data_correlation = data.corr(method='pearson')['quality'][:]             # finding correlaton with respect to quality column
sorted_data = data_correlation.sort_values(kind='quicksort', ascending=False)  #sorting values in descending order from correlated data
print("Descending order")
print(sorted_data[0:3])
print("-------------------------")

target_data = data.quality                                                  # using quality column along y-axis
features_data = data.drop(['quality'], axis=1)                              # dropping quality from data using as X_train

features_train, features_test, target_train, target_test = train_test_split(features_data, target_data, random_state=42,
                                                                            test_size=.2)

lr = linear_model.LinearRegression()                                        # fitting into liner regression model
model = lr.fit(features_train, target_train)

# Evaluate the performance
predict = model.predict(features_test)
print("R^2 is: \n", r2_score(target_test,predict))  # regression score function defines how your independent and dependent varibales are dependent on each other.

mse=mean_squared_error(target_test,predict)
rmse = math.sqrt(mse)
print('RMSE is: \n',rmse )  # root mean square error --It represents standard deviation of the differences between predicted values and observed values
