import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the data
X = pd.read_csv("larger_feature_test_set.csv")
y = pd.read_csv("larger_label_test_set.csv")

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Ensure that indices match
assert len(X) == len(y), "Mismatch between X and y lengths"

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,  # 20% of data will be used for testing
    random_state=42  # Seed for reproducibility
)
print("")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Check for missing values in y_train
print("")
print("Missing values in y_train:")
print(y_train.isna().sum())

# Drop rows in X_train and y_train where y_train has missing values
non_missing_indices = ~y_train.isna().any(axis=1)
X_train = X_train.loc[non_missing_indices]
y_train = y_train.loc[non_missing_indices]

# Check if the lengths match
assert X_train.shape[0] == y_train.shape[0], "Mismatch between X_train and y_train lengths after cleaning"
print("")
print("No Mismatch detected between y_train and X_train after cleaning.")

# Convert y_train and y_test to numeric (if needed)
# Here we assume that `y_train` and `y_test` contain categorical data and we need to encode it
# Convert categorical target data to numeric labels if necessary
y_train = y_train.apply(lambda x: x.astype('category').cat.codes)
y_test = y_test.apply(lambda x: x.astype('category').cat.codes)

# Ensure that y_train is a 1D array for classification
y_train = y_train.values.argmax(axis=1)
y_test = y_test.values.argmax(axis=1)

# Define datasets for numerical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Define datasets for categorical features
categorical_features = X.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Use sparse_output=False
])

# Combine datasets steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Print details about datasets steps
print("")
print("\nPreprocessing Details:")
print(f"Numeric Features: {numeric_features.tolist()}")
print(f"Categorical Features: {categorical_features.tolist()}")
print("\nNumeric Transformer Steps:")
print(numeric_transformer)
print("\nCategorical Transformer Steps:")
print(categorical_transformer)
print("\nColumn Transformer:")
print(preprocessor)

# Apply datasets to X_train and X_test
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Ensure the transformed shapes match
print(f"Transformed X_train shape: {X_train_transformed.shape}")
print(f"Transformed X_test shape: {X_test_transformed.shape}")

# Ensure that y_train has the correct shape after transformation
assert X_train_transformed.shape[0] == len(y_train), "Mismatch between X_train_transformed and y_train lengths"

# Define models
models = {
    'KNN': KNeighborsClassifier(),
    'RandomForest': RandomForestClassifier(class_weight='balanced')
}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"\nTraining {model_name} model...")

    # Train the model
    model.fit(X_train_transformed, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test_transformed)

    # Evaluate the model
    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


# Save the trained model to a file
joblib.dump(model, '../models/pokemon_model.pkl')

# To load the model later
model = joblib.load('../models/pokemon_model.pkl')