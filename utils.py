import pandas as pd
import os

def save_predictions_to_csv(predictions_df, filename='student_predictions.csv'):
    """Save predictions to CSV file"""
    predictions_df.to_csv(filename, index=False)
    return filename

def save_model_report(report_text, filename='model_performance_report.txt'):
    """Save model performance report to text file"""
    with open(filename, 'w') as f:
        f.write(report_text)
    return filename

def calculate_performance_category(score):
    """Categorize performance based on score"""
    if score >= 80:
        return 'High'
    elif score >= 60:
        return 'Medium'
    else:
        return 'Low'

def format_metric(value, metric_type='score'):
    """Format metric for display"""
    if metric_type == 'score':
        return f"{value:.2f}"
    elif metric_type == 'percentage':
        return f"{value:.1f}%"
    elif metric_type == 'count':
        return f"{int(value)}"
    else:
        return str(value)

def generate_summary_statistics(df):
    """Generate comprehensive summary statistics"""
    stats = {
        'total_students': len(df),
        'avg_math_score': df['math score'].mean(),
        'avg_reading_score': df['reading score'].mean(),
        'avg_writing_score': df['writing score'].mean(),
        'gender_distribution': df['gender'].value_counts().to_dict(),
        'test_prep_completion_rate': (df['test preparation course'] == 'completed').sum() / len(df) * 100
    }
    return stats
