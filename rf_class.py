import pandas as pd  # Importing pandas for data manipulation
from sklearn.ensemble import RandomForestClassifier  # Importing the RandomForestClassifier from scikit-learn
from sklearn.model_selection import train_test_split  # Importing train_test_split to split data into training and test sets

# Loading the dataset from a CSV file
mfcc = pd.read_csv("/mfcc_labeled.csv")

# Separating features (X) and target variable (y)
X = mfcc.drop('dx', axis=1)  # Dropping the target column ('dx') to keep only features
X = X.drop('File', axis=1)  # Dropping the 'File' column, assuming it's an identifier not useful for model training
y = mfcc.dx  # Extracting the target variable

# Splitting the data into training and test sets (25% test data, 75% training data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

# Initializing and training a RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)  # Training the model on the training data

# Extracting feature importance scores from the trained model
importances = rf.feature_importances_

# Creating a DataFrame to store feature names and their corresponding importance scores
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})

# Sorting features by importance in descending order
feature_importance_df.sort_values(by='Importance', ascending=False)

# Printing the feature importance rankings
print(feature_importance_df.to_string())
