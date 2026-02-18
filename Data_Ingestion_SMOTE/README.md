# Assignment: Advanced Data Ingestion and Handling Data Imbalance

## Subject
Applied Data Science

## Assignment Objective
To preprocess a credit card fraud detection dataset that has high class imbalance and apply SMOTE algorithm to balance the dataset.

---

## Dataset Details

Dataset Name: credit_card_fraud_dataset.csv

Columns:
- TransactionID
- TransactionDate
- Amount
- MerchantID
- TransactionType
- Location
- IsFraud (Target Variable)

The dataset contains more non-fraud transactions compared to fraud transactions, creating class imbalance.

---

## Tasks Performed

1. Data Ingestion using pandas
2. Data Cleaning
   - Removed duplicate records
   - Removed missing values
   - Converted Amount to numeric format
   - Dropped TransactionID column
3. Encoding categorical features using LabelEncoder
4. Feature Scaling using StandardScaler
5. Train-Test Split using stratified sampling
6. Applied SMOTE to make both classes equal
7. Trained Logistic Regression model
8. Evaluated model using confusion matrix and classification report

---

## Libraries Used

- pandas
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn

---

## Conclusion

SMOTE successfully balanced the dataset by generating synthetic samples for the minority class.  
After balancing, the model was trained and evaluated for fraud detection.

---

## Student Name
Krushna Warule  
B.Tech Computer Engineering
