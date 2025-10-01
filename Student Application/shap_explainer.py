import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class SHAPExplainer:
    def __init__(self, ml_models, preprocessor, X_train):
        self.ml_models = ml_models
        self.preprocessor = preprocessor
        self.X_train = X_train
        self.feature_importances = {}
        
        # Calculate feature importances for tree-based models
        self._calculate_importances()
    
    def _calculate_importances(self):
        """Calculate feature importances for each model"""
        # Random Forest
        rf_model = self.ml_models.get_model('random_forest')
        self.feature_importances['random_forest'] = rf_model.feature_importances_
        
        # XGBoost
        xgb_model = self.ml_models.get_model('xgboost')
        self.feature_importances['xgboost'] = xgb_model.feature_importances_
    
    def plot_feature_importance(self, model_name='random_forest'):
        """Plot feature importance"""
        importances = self.feature_importances[model_name]
        feature_names = self.preprocessor.get_feature_names()
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        # Create plotly bar chart
        fig = go.Figure(go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h',
            marker_color='#8c564b'
        ))
        
        fig.update_layout(
            title=f'Feature Importance - {model_name.replace("_", " ").title()}',
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            height=500
        )
        
        return fig
    
    def plot_shap_summary(self, model_name='random_forest'):
        """Create feature importance summary plot"""
        importances = self.feature_importances[model_name]
        feature_names = self.preprocessor.get_feature_names()
        
        # Sort by importance
        indices = np.argsort(importances)
        
        # Create matplotlib figure
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance Summary - {model_name.replace("_", " ").title()}')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_force_plot(self, model_name='random_forest', instance_idx=0):
        """Create feature contribution plot for individual prediction"""
        try:
            importances = self.feature_importances[model_name]
            feature_names = self.preprocessor.get_feature_names()
            feature_values = self.X_train.iloc[instance_idx].values
            
            # Calculate contributions (importance * feature value)
            contributions = importances * feature_values
            
            # Sort by absolute contribution
            indices = np.argsort(np.abs(contributions))[::-1][:10]  # Top 10
            
            # Create matplotlib figure
            plt.figure(figsize=(14, 6))
            colors = ['red' if contributions[i] < 0 else 'green' for i in indices]
            plt.barh(range(len(indices)), [contributions[i] for i in indices], color=colors)
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Feature Contribution')
            plt.title(f'Top Feature Contributions - Student {instance_idx}')
            plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
            plt.tight_layout()
            
            return plt.gcf()
        except Exception as e:
            print(f"Error creating force plot: {e}")
            return None
    
    def get_feature_contributions(self, model_name, instance_idx):
        """Get feature contributions for a specific instance"""
        importances = self.feature_importances[model_name]
        feature_names = self.preprocessor.get_feature_names()
        feature_values = self.X_train.iloc[instance_idx].values
        
        contributions = importances * feature_values
        
        contrib_df = pd.DataFrame({
            'feature': feature_names,
            'contribution': contributions
        }).sort_values('contribution', key=abs, ascending=False)
        
        return contrib_df
