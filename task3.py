import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv('social.csv')

# Simple keyword-based sentiment scoring
positive_words = ["love", "amazing", "great", "fantastic", "smooth", "helpful", "faster", "exceeded"]
negative_words = ["hate", "terrible", "worst", "annoying", "disappointing", "bug"]

def analyze_sentiment(post):
    post = post.lower()
    score = 0
    for word in positive_words:
        if word in post:
            score += 1
    for word in negative_words:
        if word in post:
            score -= 1
    return score

# Apply analysis
df['sentiment'] = df['post'].apply(analyze_sentiment)

# Categorize
def sentiment_category(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment_category'] = df['sentiment'].apply(sentiment_category)

# Aggregate sentiment
daily_sentiment = df.groupby('date')['sentiment'].mean().reset_index()

# Plot average sentiment
plt.figure(figsize=(10,5))
plt.plot(daily_sentiment['date'], daily_sentiment['sentiment'], marker='o')
plt.title('Average Sentiment Over Time')
plt.ylabel('Average Sentiment Score')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.ylim(-2,2)
plt.grid(True)
plt.show()

# Plot category distribution
df['sentiment_category'].value_counts().plot.pie(
    autopct='%1.1f%%', colors=['#4CAF50','#FFC107','#F44336'], startangle=140
)
plt.title('Sentiment Category Distribution')
plt.ylabel('')
plt.show()