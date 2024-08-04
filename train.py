# import libraries
import pandas as pd

# Import libraries for classification task
from sklearn.model_selection import train_test_split
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
#Remove warnings
import warnings
warnings.filterwarnings('ignore')

# Load Dataset
df = pd.read_csv('Iris.csv')

# Drop id column
df.drop('Id',axis=1,inplace=True)

X = df.drop(['Species'], axis=1)
y = df['Species']

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()

# Define the hyperparameters and their values to search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best Cross-Validation Score: {best_score}")

# Save the best model
joblib.dump(grid_search.best_estimator_, 'best_irirs_model_gridsearchcv.pkl')

print("Model dumped successfully")