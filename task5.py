import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("population.csv")
print("Dataset Preview:")
print(data.head())

plt.figure(figsize=(8, 5))
plt.hist(data['Age'], bins=8, edgecolor='black', alpha=0.7)
plt.title("Age Distribution in Population")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Gender', data=data)
plt.title("Gender Distribution in Population")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()