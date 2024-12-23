#Data Preprocessing 

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\chandra\Downloads\Data.csv')
#dataset = pd.read_csv('Path/dataset_name.csv')
dataset.isnull()
dataset.isnull().sum()

# Extracting the independent and dependent variables 
x = dataset.iloc[:, : -1].values
y = dataset.iloc[:, -1].values

# x - independent or feture variable
# y - dependent or target variable

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

# Handling missing values 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
X_train[:, 1:3]= imputer.fit_transform(X_train[:, 1:3])
X_test[:, 1:3]= imputer.transform(X_test[:, 1:3])


# Encoding categorical data
dataset.Country.unique()
dataset.Country.value_counts()
# Encoding the independent variable 
# Using OneHotEncoder

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
X_train = np.array(ct.fit_transform(X_train))
X_test = np.array(ct.transform(X_test))

#Encoding the dependent variable 
# Using LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


#Feature scaling

# Using StandardScaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

# StandardScaler: No fixed range, but values are centered around 0. 
# And typically spread within the range of [-3, 3] for normal data.

# Using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

# MinMaxScaler: Rescales values to a fixed range, typically [0, 1]

