import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class Visualizer:
    def __init__(self, df):
        self.df = df
    
    def plot_score_distribution(self):
        """Plot distribution of math scores"""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=self.df['math score'],
            nbinsx=30,
            marker_color='#1f77b4',
            name='Math Score Distribution'
        ))
        
        fig.update_layout(
            title='Distribution of Math Scores',
            xaxis_title='Math Score',
            yaxis_title='Frequency',
            height=400
        )
        
        return fig
    
    def plot_performance_by_demographics(self):
        """Plot performance across different demographic groups"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('By Gender', 'By Race/Ethnicity', 
                          'By Lunch Type', 'By Test Prep'),
            specs=[[{'type': 'box'}, {'type': 'box'}],
                   [{'type': 'box'}, {'type': 'box'}]]
        )
        
        # Gender
        for gender in self.df['gender'].unique():
            data = self.df[self.df['gender'] == gender]['math score']
            fig.add_trace(
                go.Box(y=data, name=gender.capitalize()),
                row=1, col=1
            )
        
        # Race/Ethnicity
        for race in self.df['race/ethnicity'].unique():
            data = self.df[self.df['race/ethnicity'] == race]['math score']
            fig.add_trace(
                go.Box(y=data, name=race),
                row=1, col=2
            )
        
        # Lunch Type
        for lunch in self.df['lunch'].unique():
            data = self.df[self.df['lunch'] == lunch]['math score']
            fig.add_trace(
                go.Box(y=data, name=lunch),
                row=2, col=1
            )
        
        # Test Prep
        for prep in self.df['test preparation course'].unique():
            data = self.df[self.df['test preparation course'] == prep]['math score']
            fig.add_trace(
                go.Box(y=data, name=prep),
                row=2, col=2
            )
        
        fig.update_layout(
            height=700,
            showlegend=False,
            title_text="Math Performance Across Demographics"
        )
        
        return fig
    
    def plot_parental_education_impact(self):
        """Plot impact of parental education on math scores"""
        edu_order = [
            'some high school', 'high school', 'some college',
            "associate's degree", "bachelor's degree", "master's degree"
        ]
        
        avg_scores = self.df.groupby('parental level of education')['math score'].mean()
        avg_scores = avg_scores.reindex(edu_order)
        
        fig = go.Figure(go.Bar(
            x=avg_scores.index,
            y=avg_scores.values,
            marker_color='#2ca02c',
            text=avg_scores.values.round(1),
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Average Math Score by Parental Education Level',
            xaxis_title='Parental Education',
            yaxis_title='Average Math Score',
            height=400,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def plot_test_prep_impact(self):
        """Plot impact of test preparation on scores"""
        prep_stats = self.df.groupby('test preparation course').agg({
            'math score': 'mean',
            'reading score': 'mean',
            'writing score': 'mean'
        })
        
        fig = go.Figure()
        
        subjects = ['math score', 'reading score', 'writing score']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for subject, color in zip(subjects, colors):
            fig.add_trace(go.Bar(
                x=prep_stats.index,
                y=prep_stats[subject],
                name=subject.replace(' score', '').title(),
                marker_color=color
            ))
        
        fig.update_layout(
            title='Test Preparation Impact on All Scores',
            xaxis_title='Test Preparation Course',
            yaxis_title='Average Score',
            barmode='group',
            height=400
        )
        
        return fig
    
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap of numeric features"""
        # Select numeric columns
        numeric_cols = ['math score', 'reading score', 'writing score']
        corr_matrix = self.df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(3),
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title='Score Correlation Heatmap',
            height=400
        )
        
        return fig
    
    def plot_score_comparison(self):
        """Plot comparison of all three scores"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.df['reading score'],
            y=self.df['math score'],
            mode='markers',
            name='Math vs Reading',
            marker=dict(size=5, opacity=0.6, color='#1f77b4')
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df['writing score'],
            y=self.df['math score'],
            mode='markers',
            name='Math vs Writing',
            marker=dict(size=5, opacity=0.6, color='#ff7f0e')
        ))
        
        fig.update_layout(
            title='Math Score vs Other Scores',
            xaxis_title='Score',
            yaxis_title='Math Score',
            height=400
        )
        
        return fig
