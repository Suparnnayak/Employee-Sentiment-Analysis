"""
Flight Risk Identification Module
This module identifies employees at risk of leaving based on negative message patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class FlightRiskAnalyzer:
    """
    A class to identify employees at flight risk based on negative message patterns
    """
    
    def __init__(self, df):
        """
        Initialize the flight risk analyzer
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with employee messages, dates, and sentiment labels
        """
        self.df = df.copy()
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for analysis"""
        # Ensure date column is datetime
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        elif 'timestamp' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
        
        # Filter only negative messages
        if 'sentiment' in self.df.columns:
            self.negative_df = self.df[self.df['sentiment'] == 'Negative'].copy()
        else:
            raise ValueError("DataFrame must have 'sentiment' column")
    
    def identify_flight_risks(self, threshold=4, window_days=30):
        """
        Identify employees at flight risk based on rolling 30-day window
        
        A flight risk is any employee who has sent 4 or more negative messages
        in a rolling 30-day period.
        
        Parameters:
        -----------
        threshold : int, default=4
            Minimum number of negative messages to be considered at risk
        window_days : int, default=30
            Rolling window size in days
            
        Returns:
        --------
        pd.DataFrame : DataFrame with flight risk employees and their details
        """
        if len(self.negative_df) == 0:
            return pd.DataFrame(columns=['employee', 'negative_count', 'first_negative_date', 
                                        'last_negative_date', 'risk_period_start', 'risk_period_end'])
        
        # Get employee column name
        employee_col = None
        for col in ['employee', 'employee_id', 'employee_name', 'name']:
            if col in self.negative_df.columns:
                employee_col = col
                break
        
        if employee_col is None:
            raise ValueError("Employee column not found")
        
        # Sort by date
        self.negative_df = self.negative_df.sort_values('date')
        
        flight_risks = []
        
        # For each employee, check rolling windows
        for employee in self.negative_df[employee_col].unique():
            employee_negatives = self.negative_df[
                self.negative_df[employee_col] == employee
            ].copy()
            
            # Check each negative message as a potential start of a 30-day window
            for idx, row in employee_negatives.iterrows():
                window_start = row['date']
                window_end = window_start + timedelta(days=window_days)
                
                # Count negative messages in this window
                messages_in_window = employee_negatives[
                    (employee_negatives['date'] >= window_start) &
                    (employee_negatives['date'] <= window_end)
                ]
                
                negative_count = len(messages_in_window)
                
                if negative_count >= threshold:
                    # This employee is at risk
                    flight_risks.append({
                        employee_col: employee,
                        'negative_count': negative_count,
                        'first_negative_date': messages_in_window['date'].min(),
                        'last_negative_date': messages_in_window['date'].max(),
                        'risk_period_start': window_start,
                        'risk_period_end': window_end,
                        'date_identified': row['date']
                    })
                    break  # Only need to identify once per employee
        
        if len(flight_risks) == 0:
            return pd.DataFrame(columns=[employee_col, 'negative_count', 'first_negative_date', 
                                        'last_negative_date', 'risk_period_start', 'risk_period_end'])
        
        risk_df = pd.DataFrame(flight_risks)
        
        # Remove duplicates (in case same employee appears multiple times)
        risk_df = risk_df.drop_duplicates(subset=[employee_col])
        
        # Sort by negative count (descending) and then alphabetically
        risk_df = risk_df.sort_values(['negative_count', employee_col], ascending=[False, True])
        
        return risk_df
    
    def get_flight_risk_summary(self, threshold=4, window_days=30):
        """
        Get a summary of flight risk employees
        
        Parameters:
        -----------
        threshold : int, default=4
            Minimum number of negative messages to be considered at risk
        window_days : int, default=30
            Rolling window size in days
            
        Returns:
        --------
        dict : Summary statistics about flight risks
        """
        risk_df = self.identify_flight_risks(threshold, window_days)
        
        summary = {
            'total_at_risk': len(risk_df),
            'at_risk_employees': risk_df.iloc[:, 0].tolist() if len(risk_df) > 0 else [],
            'average_negative_messages': risk_df['negative_count'].mean() if len(risk_df) > 0 else 0,
            'max_negative_messages': risk_df['negative_count'].max() if len(risk_df) > 0 else 0,
            'min_negative_messages': risk_df['negative_count'].min() if len(risk_df) > 0 else 0
        }
        
        return summary

