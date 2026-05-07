from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def create_preprocessor():

    # numerical columns
    numerical_features = [
        'Age',
        'RestingBP',
        'Cholesterol',
        'FastingBS',
        'MaxHR',
        'Oldpeak'
    ]

    # categorical columns
    categorical_features = [
        'Sex',
        'ChestPainType',
        'RestingECG',
        'ExerciseAngina',
        'ST_Slope'
    ]

    # numerical preprocessing
    numerical_transformer = Pipeline(
        steps=[
            ('scaler', StandardScaler())
        ]
    )

    # categorical preprocessing
    categorical_transformer = Pipeline(
        steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]
    )

    # combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor

# alias so tests and pipeline can use standard name
def build_preprocessor():
    return create_preprocessor()