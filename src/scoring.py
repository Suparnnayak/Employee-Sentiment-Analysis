"""
Employee Score Calculation Module
This module handles the calculation of monthly sentiment scores for employees.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class EmployeeScorer:
    """
    A class to calculate monthly sentiment scores for employees
    """
    
    def __init__(self, df):
        """
        Initialize the scorer
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with employee messages, dates, and sentiment labels
        """
        self.df = df.copy()
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for scoring"""
        # Ensure date column is datetime
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        elif 'timestamp' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
        
        # Extract year and month for grouping
        if 'date' in self.df.columns:
            self.df['year'] = self.df['date'].dt.year
            self.df['month'] = self.df['date'].dt.month
            self.df['year_month'] = self.df['date'].dt.to_period('M')
    
    def _get_message_score(self, sentiment):
        """
        Get score for a single message based on sentiment
        
        Parameters:
        -----------
        sentiment : str
            Sentiment label ('Positive', 'Negative', 'Neutral')
            
        Returns:
        --------
        int : Score for the message (+1, -1, or 0)
        """
        if sentiment == 'Positive':
            return 1
        elif sentiment == 'Negative':
            return -1
        else:  # Neutral or missing
            return 0
    
    def calculate_scores(self):
        """
        Calculate monthly sentiment scores for each employee
        
        Returns:
        --------
        pd.DataFrame : DataFrame with monthly scores per employee
        """
        if 'sentiment' not in self.df.columns:
            raise ValueError("DataFrame must have 'sentiment' column")
        
        # Get employee column name
        employee_col = None
        for col in ['employee', 'employee_id', 'employee_name', 'name']:
            if col in self.df.columns:
                employee_col = col
                break
        
        if employee_col is None:
            raise ValueError("Employee column not found. Expected: 'employee', 'employee_id', 'employee_name', or 'name'")
        
        # Add score column
        self.df['message_score'] = self.df['sentiment'].apply(self._get_message_score)
        
        # Group by employee, year, and month to calculate monthly scores
        if 'year_month' in self.df.columns:
            monthly_scores = self.df.groupby([employee_col, 'year_month'])['message_score'].sum().reset_index()
            monthly_scores.columns = [employee_col, 'year_month', 'monthly_score']
        else:
            # Fallback: group by employee, year, month
            monthly_scores = self.df.groupby([employee_col, 'year', 'month'])['message_score'].sum().reset_index()
            monthly_scores['year_month'] = pd.to_datetime(
                monthly_scores[['year', 'month']].assign(day=1)
            ).dt.to_period('M')
            monthly_scores = monthly_scores[[employee_col, 'year_month', 'message_score']]
            monthly_scores.columns = [employee_col, 'year_month', 'monthly_score']
        
        # Add additional statistics
        monthly_stats = self.df.groupby([employee_col, 'year_month']).agg({
            'message_score': ['count', 'sum'],
            'sentiment': lambda x: (x == 'Positive').sum()
        }).reset_index()
        
        monthly_stats.columns = [employee_col, 'year_month', 'message_count', 'total_score', 'positive_count']
        monthly_stats['negative_count'] = self.df.groupby([employee_col, 'year_month']).apply(
            lambda x: (x['sentiment'] == 'Negative').sum()
        ).reset_index(drop=True)
        monthly_stats['neutral_count'] = self.df.groupby([employee_col, 'year_month']).apply(
            lambda x: (x['sentiment'] == 'Neutral').sum()
        ).reset_index(drop=True)
        
        # Merge scores with stats
        result = monthly_scores.merge(monthly_stats, on=[employee_col, 'year_month'], how='left')
        
        return result
    
    def get_employee_scores(self, employee_id=None):
        """
        Get scores for a specific employee or all employees
        
        Parameters:
        -----------
        employee_id : str, optional
            Specific employee ID to filter by
            
        Returns:
        --------
        pd.DataFrame : DataFrame with scores for the specified employee(s)
        """
        scores = self.calculate_scores()
        
        if employee_id is not None:
            employee_col = scores.columns[0]  # First column should be employee
            scores = scores[scores[employee_col] == employee_id]
        
        return scores.sort_values(['year_month', 'monthly_score'], ascending=[True, False])

