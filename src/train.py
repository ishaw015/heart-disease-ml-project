import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from preprocess import create_preprocessor

# load cleaned dataset
df = pd.read_csv('../data/heart_clean.csv')

# features
X = df.drop('HeartDisease', axis=1)

# target
y = df['HeartDisease']

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# create preprocessing pipeline
preprocessor = create_preprocessor()

# models dictionary
models = {

    'Logistic Regression': LogisticRegression(
        max_iter=1000
    ),

    'Random Forest': RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    ),

    'XGBoost': XGBClassifier(
        eval_metric='logloss',
        random_state=42
    )
}

# variables to track best model
best_model = None
best_score = 0

# store results
results = []

# training loop
for name, model in models.items():

    # create pipeline
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ]
    )

    # train model
    pipeline.fit(X_train, y_train)

    # predictions
    y_pred = pipeline.predict(X_test)

    # prediction probabilities
    y_prob = pipeline.predict_proba(X_test)[:,1]

    # evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    roc_auc = roc_auc_score(y_test, y_prob)

    # save results
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    })

    # print model results
    print("\n========================")
    print(name)
    print("========================")

    print(classification_report(y_test, y_pred))

    # select best model
    if roc_auc > best_score:
        best_score = roc_auc
        best_model = pipeline
        
        
        

# create results dataframe
results_df = pd.DataFrame(results)

print("\nFINAL MODEL COMPARISON")
print(results_df)

# save results
results_df.to_csv(
    '../reports/model_results.csv',
    index=False
)

# save best model
joblib.dump(
    best_model,
    '../models/best_model.pkl'
)

print("\nBest model saved successfully!")