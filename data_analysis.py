import pandas as pd

# dataset load
df = pd.read_csv("heart.csv")

# first 5 rows
print(df.head())

print("\nShape of dataset:", df.shape)
print("\nColumns:", df.columns)

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing values:\n", df.isnull().sum())

print("\nTarget Value Counts:")
print(df['target'].value_counts())

# Features (input)
X = df.drop('target', axis=1)

# Target (output)
y = df['target']

print("X shape:", X.shape)
print("y shape:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Preprocessing Done ✅")

from sklearn.linear_model import LogisticRegression

# Model create
model = LogisticRegression()

# Model train
model.fit(X_train, y_train)

print("Model Training Complete ✅")

# Sample prediction (test data se)
sample = X_test[0].reshape(1, -1)

prediction = model.predict(sample)

print("Prediction:", prediction)

probability = model.predict_proba(sample)

print("Probability:", probability)

from sklearn.metrics import accuracy_score

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

print("Accuracy Percentage:", accuracy * 100, "%")

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

import pickle

# Save model
with open("model/heart_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully ✅")

# Save scaler
with open("model/scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

print("Scaler saved successfully ✅")