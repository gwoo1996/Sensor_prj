import numpy;
import pandas as pd;
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt


dataset_1 = pd.read_csv('../sprj/data3.csv')

y_O3 = dataset_1[['station_O3']]
x_O3_1 = dataset_1[['sensor_O3']]
x_O3_2 = dataset_1[['sensor_O3','temp_c']]
x_O3_3 = dataset_1[['sensor_O3','wind_speed']]
x_O3_4 = dataset_1[['sensor_O3','hum_perc']]
x_O3_5 = dataset_1[['sensor_O3','temp_c', 'wind_speed']]
x_O3_6 = dataset_1[['sensor_O3','temp_c', 'hum_perc']]
x_O3_7 = dataset_1[['sensor_O3','wind_speed', 'hum_perc']]
x_O3_8 = dataset_1[['sensor_O3','temp_c', 'wind_speed', 'hum_perc']]

mlr_O3_1 = LinearRegression()
mlr_O3_1.fit(x_O3_1, y_O3)
mlr_O3_2 = LinearRegression()
mlr_O3_2.fit(x_O3_2, y_O3)
mlr_O3_3 = LinearRegression()
mlr_O3_3.fit(x_O3_3, y_O3)
mlr_O3_4 = LinearRegression()
mlr_O3_4.fit(x_O3_4, y_O3)
mlr_O3_5 = LinearRegression()
mlr_O3_5.fit(x_O3_5, y_O3)
mlr_O3_6 = LinearRegression()
mlr_O3_6.fit(x_O3_6, y_O3)
mlr_O3_7 = LinearRegression()
mlr_O3_7.fit(x_O3_7, y_O3)
mlr_O3_8 = LinearRegression()
mlr_O3_8.fit(x_O3_8, y_O3)


print(mlr_O3_1.score(x_O3_1, y_O3))
print(mlr_O3_2.score(x_O3_2, y_O3))
print(mlr_O3_3.score(x_O3_3, y_O3))
print(mlr_O3_4.score(x_O3_4, y_O3))
print(mlr_O3_5.score(x_O3_5, y_O3))
print(mlr_O3_6.score(x_O3_6, y_O3))
print(mlr_O3_7.score(x_O3_7, y_O3))
print(mlr_O3_8.score(x_O3_8, y_O3))

yp_O3_1 = mlr_O3_1.predict(x_O3_1)
numpy.swapaxes(yp_O3_1, 0, 1)
dataset_1 = pd.concat([dataset_1, pd.DataFrame(yp_O3_1)], axis = 1)

print(r2_score(yp_O3_1, y_O3))

yp_O3_2 = mlr_O3_2.predict(x_O3_2)
numpy.swapaxes(yp_O3_2, 0, 1)
dataset_1 = pd.concat([dataset_1, pd.DataFrame(yp_O3_2)], axis = 1)

yp_O3_3= mlr_O3_3.predict(x_O3_3)
numpy.swapaxes(yp_O3_3, 0, 1)
dataset_1 = pd.concat([dataset_1, pd.DataFrame(yp_O3_3)], axis = 1)

yp_O3_4 = mlr_O3_4.predict(x_O3_4)
numpy.swapaxes(yp_O3_4, 0, 1)
dataset_1 = pd.concat([dataset_1, pd.DataFrame(yp_O3_4)], axis = 1)

yp_O3_5 = mlr_O3_5.predict(x_O3_5)
numpy.swapaxes(yp_O3_5, 0, 1)
dataset_1 = pd.concat([dataset_1, pd.DataFrame(yp_O3_5)], axis = 1)

yp_O3_6 = mlr_O3_6.predict(x_O3_6)
numpy.swapaxes(yp_O3_6, 0, 1)
dataset_1 = pd.concat([dataset_1, pd.DataFrame(yp_O3_6)], axis = 1)

yp_O3_7 = mlr_O3_7.predict(x_O3_7)
numpy.swapaxes(yp_O3_7, 0, 1)
dataset_1 = pd.concat([dataset_1, pd.DataFrame(yp_O3_7)], axis = 1)

yp_O3_8 = mlr_O3_8.predict(x_O3_8)
numpy.swapaxes(yp_O3_8, 0, 1)
dataset_1 = pd.concat([dataset_1, pd.DataFrame(yp_O3_8)], axis = 1)

dataset_1.to_csv("../sprj/test_O3.csv")



#x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1,train_size = 1.0, test_size = 0.0)
#x2_train, x2_test, y2_train, y2_test = train_test_split(x2,y2,train_size = 1.0, test_size = 0.0)
#y_predict = mlr.predict(x_test)
# plt.scatter(y_test, y_predict, alpha = 0.4)
# plt.xlabel("Actual")
# plt.ylabel("Predicted Rent")
# plt.show()
