import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from PIL import Image
from collections import Counter
import re

# Setup
st.set_page_config(page_title="Customer Feedback Analysis", layout="wide")
st.title("📊 Customer Feedback Analysis Dashboard")

# Load Data
df = pd.read_csv("feedback_data.csv")
sample_image = Image.open("image1.jpeg")
sample_image2 = Image.open("image2.jpeg")
# Images
st.subheader("🖼️ FLOWCHART")
col3, col4= st.columns(2)

with col3:
    st.image(sample_image, caption="Original", use_column_width=True)
with col4:
     st.image(sample_image2, caption="Original", use_column_width=True)

# Sentiment Analysis Function
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# Clean and Analyze
df.dropna(subset=['Feedback', 'Rating'], inplace=True)
df['Sentiment'] = df['Feedback'].apply(analyze_sentiment)

# Sidebar filters
st.sidebar.header("🔧 Filters")
rating_filter = st.sidebar.slider("Filter by Rating", min_value=int(df['Rating'].min()), max_value=int(df['Rating'].max()), value=(1, 5))
sentiment_filter = st.sidebar.multiselect("Filter by Sentiment", options=df['Sentiment'].unique(), default=df['Sentiment'].unique())
df_filtered = df[(df['Rating'] >= rating_filter[0]) & (df['Rating'] <= rating_filter[1]) & (df['Sentiment'].isin(sentiment_filter))]

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Sample Dataset")
    st.dataframe(df_filtered)

with col2:
    st.subheader("📊 Summary Statistics")
    st.write(df_filtered.describe(include='all'))

# KPI
avg_rating = df_filtered['Rating'].mean()
st.metric(label="⭐ Average Rating", value=round(avg_rating, 2))

# Sentiment Distribution (Pie)
st.subheader("🧠 Sentiment Distribution")
fig1, ax1 = plt.subplots()
df_filtered['Sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax1, colors=sns.color_palette('pastel'))
ax1.set_ylabel("")
st.pyplot(fig1)

# Ratings Distribution (Histogram)
st.subheader("📈 Ratings Histogram")
fig2, ax2 = plt.subplots()
sns.histplot(df_filtered['Rating'], bins=5, kde=True, ax=ax2, color='skyblue')
st.pyplot(fig2)

# Line Plot: Avg Rating Over Sample Index
st.subheader("📉 Line Plot: Ratings Over Time (by Index)")
fig3, ax3 = plt.subplots()
df_filtered['Rating'].plot(kind='line', ax=ax3, marker='o', color='green')
ax3.set_xlabel("Index")
ax3.set_ylabel("Rating")
st.pyplot(fig3)

# Scatter Plot: Rating vs. Word Count
df_filtered['Word_Count'] = df_filtered['Feedback'].apply(lambda x: len(str(x).split()))
st.subheader("🎯 Scatter Plot: Rating vs. Feedback Word Count")
fig4, ax4 = plt.subplots()
sns.scatterplot(data=df_filtered, x='Word_Count', y='Rating', hue='Sentiment', palette='Set2', ax=ax4)
st.pyplot(fig4)

# Heatmap: Correlation
st.subheader("🔁 Heatmap of Numerical Correlations")
fig5, ax5 = plt.subplots()
sns.heatmap(df_filtered[['Rating', 'Word_Count']].corr(), annot=True, cmap='coolwarm', ax=ax5)
st.pyplot(fig5)

# Frequent Words
st.subheader("💬 Frequent Words in Feedback (Top 10)")
words = " ".join(df_filtered['Feedback'].tolist()).lower()
words = re.findall(r'\b\w+\b', words)
word_freq = Counter(words)
most_common = pd.DataFrame(word_freq.most_common(10), columns=['Word', 'Frequency'])
st.dataframe(most_common)



# Footer
st.markdown("---")
st.markdown("🚀 *Built with love using Streamlit*")
