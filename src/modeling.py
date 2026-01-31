"""
Predictive Modeling Module
This module develops linear regression models to predict sentiment scores.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class SentimentPredictor:
    """
    A class to build and evaluate linear regression models for sentiment prediction
    """
    
    def __init__(self, df, monthly_scores_df):
        """
        Initialize the predictor
        
        Parameters:
        -----------
        df : pd.DataFrame
            Original DataFrame with employee messages
        monthly_scores_df : pd.DataFrame
            DataFrame with monthly scores per employee (from scoring module)
        """
        self.df = df.copy()
        self.monthly_scores_df = monthly_scores_df.copy()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def _prepare_features(self):
        """
        Prepare features for modeling
        
        Returns:
        --------
        pd.DataFrame : DataFrame with features and target variable
        """
        # Get employee column name
        employee_col = None
        for col in ['employee', 'employee_id', 'employee_name', 'name']:
            if col in self.df.columns:
                employee_col = col
                break
        
        if employee_col is None:
            raise ValueError("Employee column not found")
        
        # Get message column
        message_col = None
        for col in ['message', 'text', 'content', 'body']:
            if col in self.df.columns:
                message_col = col
                break
        
        if message_col is None:
            raise ValueError("Message column not found")
        
        # Ensure date column exists
        if 'date' not in self.df.columns:
            if 'timestamp' in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
            else:
                raise ValueError("Date column not found")
        
        # Prepare date features
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        self.df['year_month'] = self.df['date'].dt.to_period('M')
        
        # Calculate features per employee per month
        monthly_features = []
        
        for _, score_row in self.monthly_scores_df.iterrows():
            employee = score_row[employee_col]
            year_month = score_row['year_month']
            
            # Filter messages for this employee and month
            employee_month_messages = self.df[
                (self.df[employee_col] == employee) &
                (self.df['year_month'] == year_month)
            ]
            
            if len(employee_month_messages) == 0:
                continue
            
            # Calculate features
            features = {
                employee_col: employee,
                'year_month': year_month,
                'message_frequency': len(employee_month_messages),  # Number of messages in month
                'avg_message_length': employee_month_messages[message_col].astype(str).str.len().mean(),
                'total_message_length': employee_month_messages[message_col].astype(str).str.len().sum(),
                'avg_word_count': employee_month_messages[message_col].astype(str).str.split().str.len().mean(),
                'total_word_count': employee_month_messages[message_col].astype(str).str.split().str.len().sum(),
                'positive_ratio': (employee_month_messages['sentiment'] == 'Positive').sum() / len(employee_month_messages),
                'negative_ratio': (employee_month_messages['sentiment'] == 'Negative').sum() / len(employee_month_messages),
                'neutral_ratio': (employee_month_messages['sentiment'] == 'Neutral').sum() / len(employee_month_messages),
                'month': employee_month_messages['date'].dt.month.iloc[0] if len(employee_month_messages) > 0 else 0,
                'target_score': score_row['monthly_score']
            }
            
            monthly_features.append(features)
        
        features_df = pd.DataFrame(monthly_features)
        
        return features_df
    
    def build_model(self, test_size=0.2, random_state=42):
        """
        Build and train a linear regression model
        
        Parameters:
        -----------
        test_size : float, default=0.2
            Proportion of data to use for testing
        random_state : int, default=42
            Random seed for reproducibility
            
        Returns:
        --------
        dict : Model performance metrics
        """
        # Prepare features
        features_df = self._prepare_features()
        
        # Select feature columns (exclude target and identifier columns)
        exclude_cols = ['employee', 'employee_id', 'employee_name', 'name', 
                       'year_month', 'target_score']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # Remove rows with missing values
        features_df = features_df.dropna(subset=feature_cols + ['target_score'])
        
        if len(features_df) == 0:
            raise ValueError("No valid data for modeling after removing missing values")
        
        X = features_df[feature_cols].values
        y = features_df['target_score'].values
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Train model
        self.model = LinearRegression()
        self.model.fit(self.X_train_scaled, self.y_train)
        
        self.feature_names = feature_cols
        
        # Make predictions
        y_train_pred = self.model.predict(self.X_train_scaled)
        y_test_pred = self.model.predict(self.X_test_scaled)
        
        # Calculate metrics
        train_metrics = {
            'mse': mean_squared_error(self.y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'mae': mean_absolute_error(self.y_train, y_train_pred),
            'r2': r2_score(self.y_train, y_train_pred)
        }
        
        test_metrics = {
            'mse': mean_squared_error(self.y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'mae': mean_absolute_error(self.y_test, y_test_pred),
            'r2': r2_score(self.y_test, y_test_pred)
        }
        
        metrics = {
            'train': train_metrics,
            'test': test_metrics,
            'feature_names': feature_cols,
            'feature_importance': dict(zip(feature_cols, np.abs(self.model.coef_)))
        }
        
        return metrics
    
    def plot_model_performance(self, save_path=None):
        """
        Visualize model performance
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the visualization
        """
        if self.model is None:
            raise ValueError("Model must be built first. Call build_model()")
        
        y_train_pred = self.model.predict(self.X_train_scaled)
        y_test_pred = self.model.predict(self.X_test_scaled)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Training set: Actual vs Predicted
        axes[0, 0].scatter(self.y_train, y_train_pred, alpha=0.5)
        axes[0, 0].plot([self.y_train.min(), self.y_train.max()], 
                        [self.y_train.min(), self.y_train.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Score', fontsize=12)
        axes[0, 0].set_ylabel('Predicted Score', fontsize=12)
        axes[0, 0].set_title('Training Set: Actual vs Predicted', fontsize=14, fontweight='bold')
        axes[0, 0].grid(alpha=0.3)
        
        # Test set: Actual vs Predicted
        axes[0, 1].scatter(self.y_test, y_test_pred, alpha=0.5, color='orange')
        axes[0, 1].plot([self.y_test.min(), self.y_test.max()], 
                        [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Score', fontsize=12)
        axes[0, 1].set_ylabel('Predicted Score', fontsize=12)
        axes[0, 1].set_title('Test Set: Actual vs Predicted', fontsize=14, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)
        
        # Residuals plot (training)
        train_residuals = self.y_train - y_train_pred
        axes[1, 0].scatter(y_train_pred, train_residuals, alpha=0.5)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Predicted Score', fontsize=12)
        axes[1, 0].set_ylabel('Residuals', fontsize=12)
        axes[1, 0].set_title('Training Set: Residuals Plot', fontsize=14, fontweight='bold')
        axes[1, 0].grid(alpha=0.3)
        
        # Feature importance
        if self.feature_names:
            feature_importance = pd.Series(
                np.abs(self.model.coef_), 
                index=self.feature_names
            ).sort_values(ascending=False)
            
            axes[1, 1].barh(range(len(feature_importance)), feature_importance.values)
            axes[1, 1].set_yticks(range(len(feature_importance)))
            axes[1, 1].set_yticklabels(feature_importance.index)
            axes[1, 1].set_xlabel('Absolute Coefficient Value', fontsize=12)
            axes[1, 1].set_title('Feature Importance', fontsize=14, fontweight='bold')
            axes[1, 1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def get_model_summary(self):
        """
        Get a summary of the model
        
        Returns:
        --------
        dict : Model summary including coefficients and intercept
        """
        if self.model is None:
            raise ValueError("Model must be built first. Call build_model()")
        
        summary = {
            'intercept': self.model.intercept_,
            'coefficients': dict(zip(self.feature_names, self.model.coef_)),
            'n_features': len(self.feature_names),
            'n_train_samples': len(self.X_train),
            'n_test_samples': len(self.X_test)
        }
        
        return summary

