import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
df = pd.read_csv('drug_side_effects_dataset.csv')

# Data Cleaning
df = df.drop_duplicates()
df = df[(df['Age'] >= 0) & (df['Age'] <= 100)]
df = df.dropna()

# Encode categorical variables
label_encoders = {}
for column in ['Gender', 'Race', 'Side_Effect']:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    label_encoders[column] = encoder

# Split features and label
X = df[['Age', 'Gender', 'Race']]
y = df['Side_Effect']

# Scale Age
scaler = StandardScaler()
X['Age'] = scaler.fit_transform(X[['Age']])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model Training
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Model Evaluation
y_pred = classifier.predict(X_test)
print("\\nAccuracy:", accuracy_score(y_test, y_pred))
print("\\nConfusion Matrix:\\n", confusion_matrix(y_test, y_pred))
print("\\nClassification Report:\\n", classification_report(y_test, y_pred))

