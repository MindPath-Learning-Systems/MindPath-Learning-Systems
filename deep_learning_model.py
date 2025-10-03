import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class DeepLearningModel:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self._build_model()
        self.history = None
    
    def _build_model(self):
        """Build neural network architecture"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)  # Output layer for regression
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the neural network"""
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        return self.model.evaluate(X_test, y_test, verbose=0)
    
    def get_model(self):
        """Return the Keras model"""
        return self.model
