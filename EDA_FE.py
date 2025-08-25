import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('titanic')
print("Shape of data:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum())
print("\nSummary statistics:\n", df.describe(include='all'))

print("\nSurvival counts:\n", df['survived'].value_counts())
sns.histplot(df['age'].dropna(), bins=30, kde=True)
plt.title("Age Distribution")
plt.show()
sns.countplot(x='sex', hue='survived', data=df)
plt.title("Survival by Gender")
plt.show()
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

df['age'] = df['age'].fillna(df['age'].median())  
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0]) 
df['family_size'] = df['sibsp'] + df['parch'] + 1
df['age_group'] = pd.cut(df['age'],
                         bins=[0,12,18,35,60,80],
                         labels=['Child','Teen','Young Adult','Adult','Senior'])
df = pd.get_dummies(df, columns=['sex','embarked','age_group'], drop_first=True)
print("\nData after Feature Engineering:\n", df.head())
