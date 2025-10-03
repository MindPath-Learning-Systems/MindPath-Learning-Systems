import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataPreprocessor:
    def __init__(self, df):
        self.df = df.copy()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepare_data(self):
        """
        Prepare data with feature engineering and preprocessing
        """
        df = self.df.copy()
        
        # Separate features and target
        X = df.drop(['math score'], axis=1)
        y = df['math score']
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
        
        # Feature Engineering - Interaction terms
        X['avg_other_scores'] = (X['reading score'] + X['writing score']) / 2
        X['score_variance'] = np.abs(X['reading score'] - X['writing score'])
        
        # Ordinal encoding for parental education (preserve order)
        education_order = {
            0: 0,  # some high school
            1: 1,  # high school
            2: 2,  # some college
            3: 3,  # associate's degree
            4: 4,  # bachelor's degree
            5: 5   # master's degree
        }
        X['parental level of education'] = X['parental level of education'].map(education_order)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame to preserve column names
        X_train = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        X_test = pd.DataFrame(X_test_scaled, columns=self.feature_names)
        
        # Store for later use
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_new_data(self, new_df):
        """
        Preprocess new data for prediction using fitted encoders and scaler
        """
        X = new_df.copy()
        
        # Encode categorical variables using fitted encoders
        for col, le in self.label_encoders.items():
            if col in X.columns:
                X[col] = le.transform(X[col])
        
        # Feature engineering (same as training)
        X['avg_other_scores'] = (X['reading score'] + X['writing score']) / 2
        X['score_variance'] = np.abs(X['reading score'] - X['writing score'])
        
        # Ordinal encoding for parental education
        education_order = {
            0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5
        }
        X['parental level of education'] = X['parental level of education'].map(education_order)
        
        # Scale using fitted scaler
        X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=self.feature_names)
    
    def get_feature_names(self):
        """Return list of feature names"""
        return self.feature_names
