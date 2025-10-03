import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

class TimeSeriesAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
    
    def generate_time_series_data(self):
        """Generate time series representation of student data"""
        # Create synthetic time periods (academic quarters)
        n_students = len(self.df)
        
        # Generate dates over 2 academic years (8 quarters)
        start_date = datetime(2023, 1, 1)
        dates = []
        quarters = []
        
        for i in range(n_students):
            # Distribute students across quarters
            quarter = i % 8
            base_date = start_date + timedelta(days=quarter * 90)
            # Add some random days within the quarter
            random_days = np.random.randint(0, 90)
            date = base_date + timedelta(days=random_days)
            dates.append(date)
            quarters.append(f"Q{(quarter % 4) + 1} {2023 + quarter // 4}")
        
        ts_df = self.df.copy()
        ts_df['date'] = dates
        ts_df['quarter'] = quarters
        ts_df = ts_df.sort_values('date')
        
        return ts_df
    
    def plot_overall_trends(self):
        """Plot overall performance trends over time"""
        ts_df = self.generate_time_series_data()
        
        # Aggregate by month
        ts_df['month'] = pd.to_datetime(ts_df['date']).dt.to_period('M')
        monthly_avg = ts_df.groupby('month')['math score'].mean().reset_index()
        monthly_avg['month'] = monthly_avg['month'].astype(str)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_avg['month'],
            y=monthly_avg['math score'],
            mode='lines+markers',
            name='Average Math Score',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Math Score Trends Over Time',
            xaxis_title='Month',
            yaxis_title='Average Math Score',
            height=400
        )
        
        return fig
    
    def plot_gender_trends(self):
        """Plot performance trends by gender"""
        ts_df = self.generate_time_series_data()
        ts_df['month'] = pd.to_datetime(ts_df['date']).dt.to_period('M')
        
        gender_trends = ts_df.groupby(['month', 'gender'])['math score'].mean().reset_index()
        gender_trends['month'] = gender_trends['month'].astype(str)
        
        fig = px.line(
            gender_trends,
            x='month',
            y='math score',
            color='gender',
            title='Math Score Trends by Gender',
            markers=True
        )
        
        fig.update_layout(height=400)
        return fig
    
    def plot_test_prep_trends(self):
        """Plot trends for students with/without test prep"""
        ts_df = self.generate_time_series_data()
        ts_df['quarter_year'] = ts_df['quarter']
        
        prep_trends = ts_df.groupby(['quarter_year', 'test preparation course'])['math score'].mean().reset_index()
        
        fig = px.line(
            prep_trends,
            x='quarter_year',
            y='math score',
            color='test preparation course',
            title='Math Score Trends by Test Preparation',
            markers=True
        )
        
        fig.update_layout(height=400)
        return fig
    
    def plot_parental_education_trends(self):
        """Plot trends by parental education level"""
        ts_df = self.generate_time_series_data()
        ts_df['quarter_year'] = ts_df['quarter']
        
        edu_trends = ts_df.groupby(['quarter_year', 'parental level of education'])['math score'].mean().reset_index()
        
        fig = px.line(
            edu_trends,
            x='quarter_year',
            y='math score',
            color='parental level of education',
            title='Math Score Trends by Parental Education',
            markers=True
        )
        
        fig.update_layout(height=400)
        return fig
    
    def plot_lunch_trends(self):
        """Plot trends by lunch type"""
        ts_df = self.generate_time_series_data()
        ts_df['quarter_year'] = ts_df['quarter']
        
        lunch_trends = ts_df.groupby(['quarter_year', 'lunch'])['math score'].mean().reset_index()
        
        fig = px.line(
            lunch_trends,
            x='quarter_year',
            y='math score',
            color='lunch',
            title='Math Score Trends by Lunch Type',
            markers=True
        )
        
        fig.update_layout(height=400)
        return fig
    
    def plot_seasonal_patterns(self):
        """Plot seasonal patterns (quarters)"""
        ts_df = self.generate_time_series_data()
        
        # Extract quarter from quarter_year
        ts_df['quarter_num'] = ts_df['quarter'].str.extract(r'Q(\d)')[0]
        
        seasonal = ts_df.groupby('quarter_num')['math score'].agg(['mean', 'std']).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Q1', 'Q2', 'Q3', 'Q4'],
            y=seasonal['mean'],
            error_y=dict(type='data', array=seasonal['std']),
            marker_color='#2ca02c'
        ))
        
        fig.update_layout(
            title='Seasonal Performance Patterns (by Quarter)',
            xaxis_title='Academic Quarter',
            yaxis_title='Average Math Score',
            height=400
        )
        
        return fig
