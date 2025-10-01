from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np

class MLModels:
    def __init__(self):
        self.models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'xgboost': XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        }
        self.trained = False
    
    def train_all_models(self, X_train, y_train):
        """Train all ML models"""
        for name, model in self.models.items():
            model.fit(X_train, y_train)
        self.trained = True
    
    def predict(self, model_name, X):
        """Make predictions using specified model"""
        if not self.trained:
            raise ValueError("Models must be trained first")
        return self.models[model_name].predict(X)
    
    def get_model(self, model_name):
        """Get specific model"""
        return self.models.get(model_name)
    
    def get_all_models(self):
        """Get all models"""
        return self.models
