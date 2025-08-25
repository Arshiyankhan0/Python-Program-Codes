import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('titanic')
print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isna().sum())
print("\nMissing Value Percentage:\n", df.isna().mean() * 100)
plt.figure(figsize=(10,5))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()
sns.histplot(df['age'], kde=True, bins=30)
plt.title("Age Distribution (with NaNs)")
plt.show()
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['deck'] = df['deck'].cat.add_categories("Unknown").fillna("Unknown")
df = df.drop(columns=['embark_town'])
df['age_missing_flag'] = df['age'].isna().astype(int)
df['family_size'] = df['sibsp'] + df['parch'] + 1
df['age_group'] = pd.cut(df['age'],
                         bins=[0,12,18,35,60,80],
                         labels=['Child','Teen','Young Adult','Adult','Senior'])
df = pd.get_dummies(df, columns=['sex','embarked','age_group'], drop_first=True)
print("\nData After Cleaning & Feature Engineering:\n", df.head())
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
