import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score

# Load data
df = pd.read_csv('Leads.csv')

# Drop unnecessary columns
df = df.drop(['Lead Number', 'Prospect ID'], axis=1)

# Identify missing value features
features_with_NaN = [features for features in df.columns if df[features].isnull().sum() > 1]

# Handle missing values in numerical and categorical features
numerical_feat = [f for f in df.columns if df[f].isnull().sum() > 0 and df[f].dtypes != 'O']
cat_feat = [f for f in df.columns if df[f].isnull().sum() > 0 and df[f].dtypes == 'O']

for feature in numerical_feat:
    df[feature].fillna(df[feature].median(), inplace=True)

for feature in cat_feat:
    df[feature].fillna(df[feature].mode()[0], inplace=True)

# Label Encoding for binary categorical features
le = LabelEncoder()
binary_cat = [f for f in df.columns if df[f].nunique() == 2]

for f in binary_cat:
    df[f] = le.fit_transform(df[f])

# Ordinal encoding for specific features
ordinal_feat = []
for f in df.columns:
    if df[f].dtype == 'O' and df[f].nunique() <= 3:
        ordinal_feat.append(f)

df['Asymmetrique Activity Index'] = le.fit_transform(df['Asymmetrique Activity Index'])
df['Asymmetrique Profile Index'] = le.fit_transform(df['Asymmetrique Profile Index'])

# Remove low variance features
low_var_feat = [f for f in df.columns if df[f].nunique() == 1]
df.drop(columns=low_var_feat, inplace=True)

# Separate target
y = df['Converted']
X = df.drop('Converted', axis=1)

# One-hot encode remaining categorical features
remaining_cat = [f for f in X.columns if X[f].dtype == 'O']
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), remaining_cat)
    ],
    remainder='passthrough'
)
X_encoded = preprocessor.fit_transform(X)
X = pd.DataFrame(X_encoded, columns=preprocessor.get_feature_names_out())

# Remove low variance and high correlation features
low_var_feat = [f for f in X.columns if X[f].nunique() == 1]
X.drop(columns=low_var_feat, inplace=True)

corr_matrix = X.corr().abs()
upper = np.triu(corr_matrix, k=1)
to_drop = [X.columns[i] for i in range(len(X.columns)) if any(upper[i] > 0.9)]
X.drop(columns=to_drop, inplace=True)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Lead scoring
lead_scores = pd.DataFrame({
    'Lead_Index': X_test.argmax(axis=1),
    'Conversion_Probability': y_prob,
    'Actual_Converted': y_test.values
})
lead_scores = lead_scores.sort_values(by='Conversion_Probability', ascending=False)
print("\nTop 10 High-Potential Leads:")
print(lead_scores.head(10))

# Export model and test set for use in test_model.py
model = model
X_test = X_test
y_test = y_test

import joblib
joblib.dump((model, X.columns.tolist(), scaler), 'model.pkl')

