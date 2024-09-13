import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define your datasets pipeline
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(fill_value='missing', strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, ['id', 'height', 'weight', 'capture_rate', 'stats_hp', 'stats_attack', 'stats_defense', 'stats_special-attack', 'stats_special-defense', 'stats_speed']),
        ('cat', cat_transformer, ['generation', 'habitat', 'shape', 'ability_1', 'ability_2', 'ability_3'])
    ])

# Save the datasets pipeline
joblib.dump(preprocessor, 'pokemon_preprocessor.pkl')
