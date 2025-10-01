import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_student_data(n_students=1000):
    """
    Generate synthetic student performance data matching StudentsPerformance.csv structure
    """
    np.random.seed(42)
    
    # Define categories
    genders = ['female', 'male']
    races = ['group A', 'group B', 'group C', 'group D', 'group E']
    parental_education = [
        'some high school', 'high school', 'some college', 
        "associate's degree", "bachelor's degree", "master's degree"
    ]
    lunch_types = ['standard', 'free/reduced']
    test_prep = ['none', 'completed']
    
    # Generate data with realistic correlations
    data = []
    
    for _ in range(n_students):
        gender = np.random.choice(genders)
        race = np.random.choice(races)
        parent_ed = np.random.choice(parental_education, 
            p=[0.10, 0.20, 0.25, 0.20, 0.15, 0.10])  # Weighted distribution
        lunch = np.random.choice(lunch_types, p=[0.65, 0.35])
        prep = np.random.choice(test_prep, p=[0.60, 0.40])
        
        # Base scores with correlations
        base_score = np.random.normal(65, 15)
        
        # Parental education impact
        ed_bonus = {
            'some high school': -5,
            'high school': 0,
            'some college': 3,
            "associate's degree": 5,
            "bachelor's degree": 8,
            "master's degree": 12
        }[parent_ed]
        
        # Test prep impact
        prep_bonus = 5 if prep == 'completed' else 0
        
        # Lunch type impact (socioeconomic indicator)
        lunch_bonus = 3 if lunch == 'standard' else -3
        
        # Calculate scores with some randomness
        reading = max(0, min(100, base_score + ed_bonus + prep_bonus + lunch_bonus + np.random.normal(0, 5)))
        writing = max(0, min(100, base_score + ed_bonus + prep_bonus + lunch_bonus + np.random.normal(0, 5)))
        math = max(0, min(100, base_score + ed_bonus + prep_bonus + lunch_bonus + np.random.normal(0, 6)))
        
        data.append({
            'gender': gender,
            'race/ethnicity': race,
            'parental level of education': parent_ed,
            'lunch': lunch,
            'test preparation course': prep,
            'math score': int(math),
            'reading score': int(reading),
            'writing score': int(writing)
        })
    
    return pd.DataFrame(data)

def generate_student_feedback(n_feedbacks=100):
    """
    Generate sample student feedback text for NLP analysis
    """
    np.random.seed(42)
    
    positive_templates = [
        "The math class is very helpful and I'm learning a lot.",
        "I really enjoy the teaching methods used in this course.",
        "The instructor explains concepts clearly and patiently.",
        "This class has improved my understanding of mathematics significantly.",
        "Great teaching style, makes difficult topics easy to understand.",
        "I feel more confident in math after taking this class.",
        "The practice problems really help reinforce the concepts.",
        "Excellent course materials and supportive environment."
    ]
    
    negative_templates = [
        "I'm struggling to understand the material in this class.",
        "The pace of the course is too fast for me to keep up.",
        "I find the math concepts very difficult and confusing.",
        "Need more time to practice and understand the topics.",
        "The class is challenging and I need additional help.",
        "Having trouble with homework assignments and tests.",
        "Would benefit from more examples and explanations.",
        "Finding it hard to grasp the mathematical concepts."
    ]
    
    neutral_templates = [
        "The class covers standard mathematical topics.",
        "Math class is okay, neither great nor terrible.",
        "It's an average course with typical content.",
        "The class is what I expected from a math course.",
        "Standard teaching approach for mathematics.",
        "Regular math class with homework and tests."
    ]
    
    feedbacks = []
    
    for _ in range(n_feedbacks):
        sentiment_choice = np.random.choice(['positive', 'negative', 'neutral'], 
                                           p=[0.45, 0.35, 0.20])
        
        if sentiment_choice == 'positive':
            feedback = np.random.choice(positive_templates)
        elif sentiment_choice == 'negative':
            feedback = np.random.choice(negative_templates)
        else:
            feedback = np.random.choice(neutral_templates)
        
        feedbacks.append(feedback)
    
    return feedbacks
