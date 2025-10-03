import pandas as pd
import numpy as np
import re

class PerformanceChatbot:
    def __init__(self, df):
        self.df = df
        self.responses = self._initialize_responses()
    
    def _initialize_responses(self):
        """Initialize response patterns and handlers"""
        return {
            'average_score': {
                'patterns': ['average', 'mean', 'avg'],
                'keywords': ['math', 'score'],
                'handler': self._get_average_score
            },
            'gender_impact': {
                'patterns': ['gender', 'male', 'female', 'boys', 'girls'],
                'keywords': ['affect', 'impact', 'difference', 'perform'],
                'handler': self._get_gender_analysis
            },
            'test_prep': {
                'patterns': ['test prep', 'preparation', 'prep course'],
                'keywords': ['help', 'impact', 'effect', 'benefit'],
                'handler': self._get_test_prep_analysis
            },
            'parental_education': {
                'patterns': ['parent', 'parental', 'education', 'degree'],
                'keywords': ['impact', 'effect', 'influence'],
                'handler': self._get_parental_education_analysis
            },
            'count': {
                'patterns': ['how many', 'number of', 'count', 'total'],
                'keywords': ['student'],
                'handler': self._get_student_count
            },
            'lunch': {
                'patterns': ['lunch', 'meal'],
                'keywords': ['standard', 'free', 'reduced', 'statistics', 'stats'],
                'handler': self._get_lunch_analysis
            },
            'race': {
                'patterns': ['race', 'ethnicity', 'group'],
                'keywords': ['perform', 'compare', 'difference'],
                'handler': self._get_race_analysis
            },
            'correlation': {
                'patterns': ['correlation', 'relationship', 'relate'],
                'keywords': ['reading', 'writing', 'math'],
                'handler': self._get_correlation_analysis
            }
        }
    
    def get_response(self, user_input):
        """Generate response based on user input"""
        user_input = user_input.lower()
        
        # Check each response pattern
        for response_type, config in self.responses.items():
            # Check if any pattern matches
            pattern_match = any(pattern in user_input for pattern in config['patterns'])
            keyword_match = any(keyword in user_input for keyword in config['keywords'])
            
            if pattern_match and keyword_match:
                return config['handler']()
        
        # Default response
        return self._get_default_response()
    
    def _get_average_score(self):
        """Get average math score"""
        avg_score = self.df['math score'].mean()
        return f"The average math score across all students is {avg_score:.2f}. The scores range from {self.df['math score'].min()} to {self.df['math score'].max()}."
    
    def _get_gender_analysis(self):
        """Analyze gender impact on performance"""
        gender_stats = self.df.groupby('gender')['math score'].agg(['mean', 'count'])
        
        response = "Here's the gender-based performance analysis:\n\n"
        for gender, stats in gender_stats.iterrows():
            response += f"- {gender.capitalize()}: Average score = {stats['mean']:.2f} ({int(stats['count'])} students)\n"
        
        diff = abs(gender_stats.loc['female', 'mean'] - gender_stats.loc['male', 'mean'])
        response += f"\nThe difference in average scores is {diff:.2f} points."
        
        return response
    
    def _get_test_prep_analysis(self):
        """Analyze test preparation impact"""
        prep_stats = self.df.groupby('test preparation course')['math score'].agg(['mean', 'count'])
        
        response = "Test Preparation Course Impact:\n\n"
        for prep, stats in prep_stats.iterrows():
            response += f"- {prep.capitalize()}: Average score = {stats['mean']:.2f} ({int(stats['count'])} students)\n"
        
        if 'completed' in prep_stats.index and 'none' in prep_stats.index:
            diff = prep_stats.loc['completed', 'mean'] - prep_stats.loc['none', 'mean']
            response += f"\nStudents who completed test preparation scored {diff:.2f} points higher on average."
        
        return response
    
    def _get_parental_education_analysis(self):
        """Analyze parental education impact"""
        edu_stats = self.df.groupby('parental level of education')['math score'].mean().sort_values(ascending=False)
        
        response = "Impact of Parental Education on Math Scores:\n\n"
        for edu, score in edu_stats.items():
            response += f"- {edu.title()}: {score:.2f}\n"
        
        response += f"\nHighest average: {edu_stats.index[0].title()} ({edu_stats.iloc[0]:.2f})"
        response += f"\nLowest average: {edu_stats.index[-1].title()} ({edu_stats.iloc[-1]:.2f})"
        
        return response
    
    def _get_student_count(self):
        """Get total student count and demographics"""
        total = len(self.df)
        gender_counts = self.df['gender'].value_counts()
        
        response = f"Dataset contains {total} students.\n\n"
        response += "Demographics:\n"
        for gender, count in gender_counts.items():
            pct = (count / total) * 100
            response += f"- {gender.capitalize()}: {count} ({pct:.1f}%)\n"
        
        return response
    
    def _get_lunch_analysis(self):
        """Analyze lunch type statistics"""
        lunch_stats = self.df.groupby('lunch').agg({
            'math score': ['mean', 'count']
        })
        
        response = "Lunch Type Analysis:\n\n"
        for lunch in lunch_stats.index:
            mean_score = lunch_stats.loc[lunch, ('math score', 'mean')]
            count = lunch_stats.loc[lunch, ('math score', 'count')]
            response += f"- {lunch.title()}: {int(count)} students, avg score = {mean_score:.2f}\n"
        
        return response
    
    def _get_race_analysis(self):
        """Analyze performance by race/ethnicity"""
        race_stats = self.df.groupby('race/ethnicity')['math score'].mean().sort_values(ascending=False)
        
        response = "Performance by Race/Ethnicity:\n\n"
        for race, score in race_stats.items():
            response += f"- {race.title()}: {score:.2f}\n"
        
        return response
    
    def _get_correlation_analysis(self):
        """Analyze correlation between scores"""
        corr_math_reading = self.df['math score'].corr(self.df['reading score'])
        corr_math_writing = self.df['math score'].corr(self.df['writing score'])
        
        response = "Score Correlations:\n\n"
        response += f"- Math & Reading: {corr_math_reading:.3f}\n"
        response += f"- Math & Writing: {corr_math_writing:.3f}\n\n"
        
        if corr_math_reading > 0.7:
            response += "Strong positive correlation between math and reading scores."
        elif corr_math_reading > 0.4:
            response += "Moderate positive correlation between math and reading scores."
        
        return response
    
    def _get_default_response(self):
        """Default response for unrecognized queries"""
        return """I can help you analyze student performance data. Try asking:
        
- What is the average math score?
- How does gender affect performance?
- Does test preparation help?
- What is the impact of parental education?
- How many students are in the dataset?
- Show me statistics about lunch types
- Compare performance by race/ethnicity

What would you like to know?"""
