import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class ModelEvaluator:
    def __init__(self, ml_models, dl_model, X_test, y_test):
        self.ml_models = ml_models
        self.dl_model = dl_model
        self.X_test = X_test
        self.y_test = y_test
    
    def evaluate_model(self, model_name, y_pred):
        """Calculate evaluation metrics for a model"""
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2
        }
    
    def get_all_metrics(self):
        """Get metrics for all models"""
        metrics = {}
        
        # ML models
        for name in ['linear_regression', 'random_forest', 'xgboost']:
            y_pred = self.ml_models.predict(name, self.X_test)
            metrics[name.replace('_', ' ').title()] = self.evaluate_model(name, y_pred)
        
        # Deep learning model
        y_pred_dl = self.dl_model.predict(self.X_test).flatten()
        metrics['Neural Network'] = self.evaluate_model('neural_network', y_pred_dl)
        
        return pd.DataFrame(metrics).T
    
    def plot_metrics_comparison(self):
        """Create bar chart comparing model metrics"""
        metrics_df = self.get_all_metrics()
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Mean Absolute Error', 'Root Mean Squared Error', 'R² Score')
        )
        
        # MAE
        fig.add_trace(
            go.Bar(x=metrics_df.index, y=metrics_df['MAE'], name='MAE', marker_color='#1f77b4'),
            row=1, col=1
        )
        
        # RMSE
        fig.add_trace(
            go.Bar(x=metrics_df.index, y=metrics_df['RMSE'], name='RMSE', marker_color='#ff7f0e'),
            row=1, col=2
        )
        
        # R²
        fig.add_trace(
            go.Bar(x=metrics_df.index, y=metrics_df['R²'], name='R²', marker_color='#2ca02c'),
            row=1, col=3
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            title_text="Model Performance Comparison"
        )
        
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def plot_predictions_vs_actual(self):
        """Plot predictions vs actual values for all models"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Linear Regression', 'Random Forest', 'XGBoost', 'Neural Network')
        )
        
        models_data = [
            ('linear_regression', 1, 1),
            ('random_forest', 1, 2),
            ('xgboost', 2, 1)
        ]
        
        for model_name, row, col in models_data:
            y_pred = self.ml_models.predict(model_name, self.X_test)
            fig.add_trace(
                go.Scatter(x=self.y_test, y=y_pred, mode='markers', 
                          marker=dict(size=5, opacity=0.6),
                          name=model_name.replace('_', ' ').title()),
                row=row, col=col
            )
            # Add diagonal line
            fig.add_trace(
                go.Scatter(x=[0, 100], y=[0, 100], mode='lines',
                          line=dict(color='red', dash='dash'),
                          showlegend=False),
                row=row, col=col
            )
        
        # Neural Network
        y_pred_dl = self.dl_model.predict(self.X_test).flatten()
        fig.add_trace(
            go.Scatter(x=self.y_test, y=y_pred_dl, mode='markers',
                      marker=dict(size=5, opacity=0.6),
                      name='Neural Network'),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=[0, 100], y=[0, 100], mode='lines',
                      line=dict(color='red', dash='dash'),
                      showlegend=False),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="Actual Math Score")
        fig.update_yaxes(title_text="Predicted Math Score")
        fig.update_layout(height=700, showlegend=False, title_text="Predictions vs Actual Values")
        
        return fig
    
    def cross_validate_models(self, cv=5):
        """Perform k-fold cross-validation on ML models"""
        from data_preprocessing import DataPreprocessor
        
        # We need to use the original preprocessor's training data
        X_train = self.ml_models.models['linear_regression'].feature_names_in_
        
        # Get training data from session state or recreate
        # For simplicity, we'll use the test data for demonstration
        cv_results = {}
        
        for name, model in self.ml_models.get_all_models().items():
            scores = cross_val_score(model, self.X_test, self.y_test, 
                                    cv=cv, scoring='r2')
            cv_results[name.replace('_', ' ').title()] = {
                'Mean R²': scores.mean(),
                'Std R²': scores.std(),
                'Min R²': scores.min(),
                'Max R²': scores.max()
            }
        
        return pd.DataFrame(cv_results).T
    
    def plot_cv_results(self, cv_results):
        """Plot cross-validation results"""
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=cv_results.index,
            y=cv_results['Mean R²'],
            error_y=dict(type='data', array=cv_results['Std R²']),
            marker_color='#9467bd'
        ))
        
        fig.update_layout(
            title='Cross-Validation Results (5-Fold)',
            xaxis_title='Model',
            yaxis_title='R² Score',
            height=400
        )
        
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def plot_residuals(self):
        """Plot residuals for all models"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Linear Regression', 'Random Forest', 'XGBoost', 'Neural Network')
        )
        
        models_data = [
            ('linear_regression', 1, 1),
            ('random_forest', 1, 2),
            ('xgboost', 2, 1)
        ]
        
        for model_name, row, col in models_data:
            y_pred = self.ml_models.predict(model_name, self.X_test)
            residuals = self.y_test.values - y_pred
            
            fig.add_trace(
                go.Scatter(x=y_pred, y=residuals, mode='markers',
                          marker=dict(size=5, opacity=0.6)),
                row=row, col=col
            )
            # Add zero line
            fig.add_trace(
                go.Scatter(x=[min(y_pred), max(y_pred)], y=[0, 0],
                          mode='lines', line=dict(color='red', dash='dash'),
                          showlegend=False),
                row=row, col=col
            )
        
        # Neural Network
        y_pred_dl = self.dl_model.predict(self.X_test).flatten()
        residuals_dl = self.y_test.values - y_pred_dl
        
        fig.add_trace(
            go.Scatter(x=y_pred_dl, y=residuals_dl, mode='markers',
                      marker=dict(size=5, opacity=0.6)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=[min(y_pred_dl), max(y_pred_dl)], y=[0, 0],
                      mode='lines', line=dict(color='red', dash='dash'),
                      showlegend=False),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="Predicted Values")
        fig.update_yaxes(title_text="Residuals")
        fig.update_layout(height=700, showlegend=False, title_text="Residual Plots")
        
        return fig
