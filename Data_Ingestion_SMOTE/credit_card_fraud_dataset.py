# ============================================================
# Assignment: Advanced Data Ingestion & Handling Data Imbalance
# Dataset: credit_card_fraud_dataset.csv
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE


# -------------------- Data Ingestion --------------------
df = pd.read_csv("credit_card_fraud_dataset.csv")


# -------------------- Data Cleaning --------------------
df.drop_duplicates(inplace=True)
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
df.dropna(inplace=True)
df.drop(columns=['TransactionID'], inplace=True)


# -------------------- Encoding --------------------
le = LabelEncoder()
df['TransactionDate'] = le.fit_transform(df['TransactionDate'])
df['TransactionType'] = le.fit_transform(df['TransactionType'])
df['Location'] = le.fit_transform(df['Location'])


# -------------------- Check Class Distribution --------------------
print("Before SMOTE:\n", df['IsFraud'].value_counts())

plt.figure()
sns.countplot(x='IsFraud', data=df)
plt.title("Before SMOTE")
plt.show()


# -------------------- Feature & Target --------------------
X = df.drop('IsFraud', axis=1)
y = df['IsFraud']


# -------------------- Train Test Split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# -------------------- Scaling --------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -------------------- SMOTE --------------------
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:\n", pd.Series(y_train_resampled).value_counts())

plt.figure()
sns.countplot(x=y_train_resampled)
plt.title("After SMOTE")
plt.show()


# -------------------- Model Training --------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_resampled, y_train_resampled)


# -------------------- Prediction & Evaluation --------------------
y_pred = model.predict(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\nAssignment Completed Successfully")
