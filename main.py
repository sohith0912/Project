# import all the required libaries
import  pandas as pd
import numpy as np
import matplotlib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random

# loading datasets
dataset = pd.read_csv("./car_price.csv", index_col=0)

# top-5 rows
display(dataset.head())


# function to process the data
def process(row):
    temp = pd.DataFrame()
    carName = row['car_name'].split()[:2]
    temp['Car Brand'] = [carName[0]]
    temp['Car Model'] = [carName[1]]
    price = row['car_prices_in_rupee']
    if "Lakh" in price:
        price = float(price.strip("Lakh")) * 100000
    elif "Crore" in price:
        price = float(price.strip("Crore")) * 1000000
    else:
        price = float(price.replace(",", ""))
    temp['Price'] = [round(price, 2)]
    temp['KM'] = [
        round(float(row['kms_driven'].strip("kms").replace(",", "")), 2)
    ]
    temp['Engine'] = [row['engine'].strip("cc")]
    temp['Seats'] = [row['Seats'].strip('Seats')]
    temp['Fuel'] = [row['fuel_type']]
    temp['Transmission'] = [row['transmission']]
    temp['Ownership'] = [row['ownership']]
    temp['Year'] = [int(row['manufacture'])]
    return temp


# data preprocessing
cleanedDataDF = pd.DataFrame(columns=[
    'Car Brand', 'Car Model', 'Price', 'KM', 'Engine', 'Seats', 'Fuel',
    'Transmission', 'Ownership', 'Year'
])
for _, row in dataset.iterrows():
    cleanedData = process(row)
    cleanedDataDF = pd.concat([cleanedDataDF, cleanedData], axis=0)

# displaying the top-5 rows after preprocessing
display(cleanedDataDF.head())

# displaying the diemension of the dataset
print(f"The dataset has {cleanedDataDF.shape[0]} rows and {cleanedDataDF.shape[1]} columns post preprocessing")

# EDA of the dataset
print("Statistical Analysis of the dataset")
cleanedDataDF[['Engine', 'Seats', 'Year']] = cleanedDataDF[['Engine', 'Seats', 'Year']].astype(int)
display(cleanedDataDF.describe())

print("Information about the data types of the dataset")
display(cleanedDataDF.info())

# data visualisation
fig, axes = plt.subplots(1,2, figsize=(21,6))
sns.distplot(cleanedDataDF['Price'], ax=axes[0])
sns.distplot(np.log1p(cleanedDataDF['Price']), ax=axes[1])
axes[1].set_xlabel('log(1+price)')
plt.show()

# car count based on manufacturer
ax = sns.countplot(data=cleanedDataDF, x=cleanedDataDF['Car Brand'])
ax.tick_params(axis='x', rotation=90)
plt.show()

# car count based on ownership
ax = sns.countplot(data=cleanedDataDF, x=cleanedDataDF['Ownership'])
ax.tick_params(axis='x', rotation=90)
plt.show()

# car count based on year
ax = sns.countplot(data=cleanedDataDF, x=cleanedDataDF['Year'])
ax.tick_params(axis='x', rotation=90)
plt.show()

# distribution of cars by transmission
labels = ['Manual', 'Automatic']
plt.pie(cleanedDataDF['Transmission'].value_counts(), labels = labels, autopct='%.0f%%')
plt.legend()
plt.show()

# distribution of cars by fuel type
labels = ['Petrol', 'Diesel', 'Cng', 'Lpg', 'Electric']
plt.pie(cleanedDataDF['Fuel'].value_counts(), labels = labels, explode=[0, 0, 1, 1, 1], autopct='%.0f%%')
plt.legend()
plt.show()

vis_1=pd.pivot_table(cleanedDataDF, index=['Year'],values = ['Price'],aggfunc = 'mean')
vis_1.plot(kind='line',linewidth=4.5,figsize=(12,7),title='Average car price by Year')
plt.show()

plt.figure(figsize=(10,7))
sns.heatmap(cleanedDataDF.corr(), annot=True,linewidths=.5,fmt='.2f')
plt.title("Correlation Graph",size=18)
plt.show()

sns.pairplot(cleanedDataDF)
plt.show()

# Converting category into numerical
labelencoder = LabelEncoder()
for col in cleanedDataDF.loc[:, cleanedDataDF.dtypes == 'O'].columns:
    cleanedDataDF[col] = labelencoder.fit_transform(cleanedDataDF[col])

# splitting dataset
x = cleanedDataDF.drop(columns="Price")
y = cleanedDataDF.Price
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

print("Length of the datasets")
print(x.shape, xtrain.shape, xtest.shape)

# defining model
# Create MLP model
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', alpha=0.001, max_iter=1000)

# Train model on training data
mlp.fit(xtrain, ytrain)

# Predict prices on testing data
ypred = mlp.predict(xtest)

# Evaluate model performance
mse = mean_squared_error(ytest, ypred)
r2 = r2_score(ytest, ypred)

print(f'Mean squared error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')


# Create KNN Regression model
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', alpha=0.001, max_iter=1000)

# Train model on training data
mlp.fit(xtrain, ytrain)

# Predict prices on testing data
ypred = mlp.predict(xtest)

# Evaluate model performance
mse = mean_squared_error(ytest, ypred)
r2 = r2_score(ytest, ypred)

print(f'Mean squared error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# Visualize predicted vs actual values
plt.scatter(ytest, ypred)
plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Predicted vs Actual Prices of MLP')
plt.show()

# Create KNN model
knn = KNeighborsRegressor(n_neighbors=5)

# Train model on training data
knn.fit(xtrain, ytrain)

# Predict prices on testing data
ypred = knn.predict(xtest)

# Evaluate model performance
mse = mean_squared_error(ytest, ypred)
r2 = r2_score(ytest, ypred)

print(f'Mean squared error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# Visualize predicted vs actual values
plt.scatter(ytest, ypred)
plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Predicted vs Actual Prices of KNN')
plt.show()
