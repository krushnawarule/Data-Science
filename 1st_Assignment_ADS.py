# =========================================================
# STUDENT DATA PREPROCESSING & VISUALIZATION
# =========================================================

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 2. Load the dataset
# ---------------------------------------------------------
df = pd.read_csv("student-scores.csv")
print("Dataset loaded successfully")
print("=" * 100)

# ---------------------------------------------------------
# 3. Clean column names 
# ---------------------------------------------------------
df.columns = (df.columns.str.strip().str.lower())

print("Column names after cleaning:")
print(df.columns.tolist())
print("=" * 100)

# ---------------------------------------------------------
# 4. Dataset exploration
# ---------------------------------------------------------
print("First 5 records:")
print(df.head())
print("=" * 100)

print("\nDataset shape (rows, columns):")
print(df.shape)
print("=" * 100)

print("\nDataset information:")
df.info()
print("=" * 100)

# ---------------------------------------------------------
# 5. Check missing values and zero values
# ---------------------------------------------------------
print("Missing values in each column:")
print(df.isnull().sum())
print("=" * 100)

numerical_cols = df.select_dtypes(include=np.number).columns
print("\nZero values in numerical columns:")
print((df[numerical_cols] == 0).sum())
print("=" * 100)

# ---------------------------------------------------------
# 6. Remove duplicate records
# ---------------------------------------------------------
print("Duplicate records:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Duplicates removed")
print("=" * 100)

# ---------------------------------------------------------
# 7. Handle missing values
# ---------------------------------------------------------

# Fill numerical columns with MEAN
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# Fill categorical columns with MODE
cat_cols = df.select_dtypes(include=['object', 'string', 'bool']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values handled")
print("=" * 100)

# ---------------------------------------------------------
# 8. Detect subject score columns automatically
# ---------------------------------------------------------
score_cols = [col for col in df.columns if col.endswith("_score")]
print("Detected subject score columns:", score_cols)

# Ensure scores are numeric
df[score_cols] = df[score_cols].apply(pd.to_numeric, errors='coerce')
df[score_cols] = df[score_cols].fillna(df[score_cols].mean())
print("=" * 100)

# ---------------------------------------------------------
# 9. Calculate TOTAL SCORE and PERCENTAGE
# ---------------------------------------------------------
df['total_score'] = df[score_cols].sum(axis=1)

# Percentage is used ONLY for bar graph
df['percentage'] = (df['total_score'] / (len(score_cols) * 100)) * 100

print("Total score and percentage calculated")
print("=" * 100)

# ---------------------------------------------------------
# 10. Statistical measures (TOTAL SCORE)
# ---------------------------------------------------------
print("Total Score Statistics")
print("Mean:", df['total_score'].mean())
print("Median:", df['total_score'].median())
print("Mode:", df['total_score'].mode()[0])
print("Skewness:", df['total_score'].skew())
print("=" * 100)

# ---------------------------------------------------------
# 11. Bar graph – Top 5 students (PERCENTAGE)
# ---------------------------------------------------------
top5 = df.sort_values(by='percentage', ascending=False).head(5)

plt.figure()
plt.bar(top5['first_name'], top5['percentage'])
plt.xlabel("Student Name")
plt.ylabel("Overall Percentage")
plt.title("Top 5 Students by Percentage")
plt.show()

# ---------------------------------------------------------
# 12. Average marks in each subject
# ---------------------------------------------------------
plt.figure()
df[score_cols].mean().plot(kind='bar')
plt.ylabel("Average Marks")
plt.title("Average Marks in Each Subject")
plt.show()

# ---------------------------------------------------------
# 13. Pie chart – Part-time job distribution
# ---------------------------------------------------------
plt.figure()
df['part_time_job'].value_counts().plot(
    kind='pie', autopct='%1.1f%%'
)
plt.title("Part-Time Job Distribution")
plt.ylabel("")
plt.show()

# ---------------------------------------------------------
# 14. Scatter plot – Study hours vs TOTAL SCORE
# ---------------------------------------------------------
plt.figure()
plt.scatter(df['weekly_self_study_hours'], df['total_score'])
plt.xlabel("Weekly Self Study Hours")
plt.ylabel("Total Score")
plt.title("Study Hours vs Total Score")
plt.show()

# ---------------------------------------------------------
# 15. Correlation heatmap (TOTAL SCORE only)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.heatmap(
    df[score_cols + ['weekly_self_study_hours', 'absence_days', 'total_score']].corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Heatmap (Using Total Score)")
plt.show()

# ---------------------------------------------------------
# 16. Boxplot – Absence days vs TOTAL SCORE
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['absence_days'], y=df['total_score'])
plt.xlabel("Absence Days")
plt.ylabel("Total Score")
plt.title("Absence Days vs Total Score")
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(x='absence_days', data=df)
plt.xlabel("Absence Days")
plt.ylabel("Number of Students")
plt.title("Distribution of Students by Absence Days")
plt.show()

plt.figure(figsize=(8, 5))
sns.regplot(
    x='weekly_self_study_hours',
    y='total_score',
    data=df
)
plt.xlabel("Weekly Self Study Hours")
plt.ylabel("Total Score")
plt.title("Self Study Hours vs Total Score (Regression)")
plt.show()

print(" preprocessing tasks completed")
print("=" * 100)

# END 

