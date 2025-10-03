import pandas as pd
import numpy as np
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from data_generator import generate_student_feedback

class NLPAnalyzer:
    def __init__(self):
        pass
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text using TextBlob"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Classify sentiment
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return sentiment, polarity, subjectivity
    
    def generate_sample_feedback(self, n_samples=100):
        """Generate sample student feedback with sentiment analysis"""
        feedbacks = generate_student_feedback(n_samples)
        
        data = []
        for feedback in feedbacks:
            sentiment, polarity, subjectivity = self.analyze_sentiment(feedback)
            
            # Assign random performance level
            if sentiment == 'positive':
                performance = np.random.choice(['High', 'Medium', 'Low'], p=[0.6, 0.3, 0.1])
            elif sentiment == 'negative':
                performance = np.random.choice(['High', 'Medium', 'Low'], p=[0.1, 0.3, 0.6])
            else:
                performance = np.random.choice(['High', 'Medium', 'Low'], p=[0.3, 0.4, 0.3])
            
            data.append({
                'feedback': feedback,
                'sentiment': sentiment,
                'polarity': polarity,
                'subjectivity': subjectivity,
                'performance_level': performance
            })
        
        return pd.DataFrame(data)
    
    def plot_sentiment_distribution(self, feedback_df):
        """Plot sentiment distribution"""
        sentiment_counts = feedback_df['sentiment'].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                marker_color=['#2ca02c', '#d62728', '#1f77b4']
            )
        ])
        
        fig.update_layout(
            title='Student Feedback Sentiment Distribution',
            xaxis_title='Sentiment',
            yaxis_title='Count',
            height=400
        )
        
        return fig
    
    def plot_sentiment_by_performance(self, feedback_df):
        """Plot sentiment distribution by performance level"""
        grouped = feedback_df.groupby(['performance_level', 'sentiment']).size().reset_index(name='count')
        
        fig = px.bar(
            grouped,
            x='performance_level',
            y='count',
            color='sentiment',
            title='Sentiment Distribution by Performance Level',
            barmode='group',
            color_discrete_map={
                'positive': '#2ca02c',
                'negative': '#d62728',
                'neutral': '#1f77b4'
            }
        )
        
        fig.update_layout(height=400)
        return fig
    
    def create_wordcloud(self, feedback_df, sentiment_filter=None):
        """Create word cloud from feedback text"""
        if sentiment_filter:
            texts = feedback_df[feedback_df['sentiment'] == sentiment_filter]['feedback'].tolist()
        else:
            texts = feedback_df['feedback'].tolist()
        
        text = ' '.join(texts)
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis'
        ).generate(text)
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud - {sentiment_filter.title() if sentiment_filter else "All"} Feedback', 
                 fontsize=16)
        plt.tight_layout()
        
        return plt.gcf()
    
    def get_polarity_stats(self, feedback_df):
        """Get polarity statistics"""
        stats = {
            'mean_polarity': feedback_df['polarity'].mean(),
            'median_polarity': feedback_df['polarity'].median(),
            'std_polarity': feedback_df['polarity'].std(),
            'mean_subjectivity': feedback_df['subjectivity'].mean()
        }
        return stats
