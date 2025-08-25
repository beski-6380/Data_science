import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('traffic.csv')

# Convert datetime column and extract hour for time of day analysis
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
df['hour'] = df['datetime'].dt.hour

# Categorize time of day
def time_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

df['time_of_day'] = df['hour'].apply(time_of_day)

# Group by weather, road condition, and time of day to count accidents
accidents_summary = df.groupby(['weather', 'road_condition', 'time_of_day']).size().reset_index(name='count')

# --- Visualization 1: Accident counts by weather ---
plt.figure(figsize=(8, 5))
sns.barplot(data=accidents_summary, x='weather', y='count', estimator=sum)
plt.title('Accident Count by Weather')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Visualization 2: Accident counts by road condition ---
plt.figure(figsize=(8, 5))
sns.barplot(data=accidents_summary, x='road_condition', y='count', estimator=sum)
plt.title('Accident Count by Road Condition')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Visualization 3: Accident counts by time of day ---
plt.figure(figsize=(8, 5))
sns.barplot(data=accidents_summary, x='time_of_day', y='count', estimator=sum,
            order=['Morning', 'Afternoon', 'Evening', 'Night'])
plt.title('Accident Count by Time of Day')
plt.tight_layout()
plt.show()

# --- Visualization 4: Hotspot scatter plot (without GeoPandas) ---
plt.figure(figsize=(8, 6))
plt.scatter(df['longitude'], df['latitude'], c='red', alpha=0.5, s=30)
plt.title('Traffic Accident Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.tight_layout()
plt.show()