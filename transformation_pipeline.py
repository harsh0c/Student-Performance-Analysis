import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle

# Load the dataset
data = pd.read_csv('stud.csv')

# Define feature columns and target column
feature_columns = [
    'gender', 'race_ethnicity', 'parental_level_of_education',
    'lunch', 'test_preparation_course', 'reading_score', 'writing_score'
]
target_column = 'math_score'

# Separate features and target
X = data[feature_columns]
y = data[target_column]

# Ensure X is a DataFrame
if not isinstance(X, pd.DataFrame):
    X = pd.DataFrame(X, columns=feature_columns)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing pipeline
categorical_features = [
    'gender', 'race_ethnicity', 'parental_level_of_education',
    'lunch', 'test_preparation_course'
]
numerical_features = ['reading_score', 'writing_score']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ]
)

# Create the full pipeline with preprocessing and linear regression model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', LinearRegression())])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Save the pipeline to a pickle file
with open('pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
