"""
Employee Ranking Module
This module handles ranking of employees based on their monthly sentiment scores.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class EmployeeRanker:
    """
    A class to rank employees based on their monthly sentiment scores
    """
    
    def __init__(self, monthly_scores_df):
        """
        Initialize the ranker
        
        Parameters:
        -----------
        monthly_scores_df : pd.DataFrame
            DataFrame with monthly scores per employee (from scoring module)
        """
        self.monthly_scores = monthly_scores_df.copy()
    
    def get_top_employees(self, top_n=3, positive=True, year_month=None):
        """
        Get top N employees per month based on sentiment scores
        
        Parameters:
        -----------
        top_n : int, default=3
            Number of top employees to return
        positive : bool, default=True
            If True, return top positive; if False, return top negative
        year_month : str or Period, optional
            Specific month to filter by. If None, returns for all months
            
        Returns:
        --------
        pd.DataFrame : DataFrame with top employees
        """
        employee_col = self.monthly_scores.columns[0]
        results = []
        
        # Filter by month if specified
        if year_month is not None:
            months_to_process = [year_month]
        else:
            months_to_process = self.monthly_scores['year_month'].unique()
        
        for ym in months_to_process:
            month_data = self.monthly_scores[
                self.monthly_scores['year_month'] == ym
            ].copy()
            
            if len(month_data) == 0:
                continue
            
            if positive:
                # Top positive (highest scores)
                top_employees = month_data.nlargest(top_n, 'monthly_score')
            else:
                # Top negative (lowest scores)
                top_employees = month_data.nsmallest(top_n, 'monthly_score')
            
            # Sort by score (descending) then alphabetically
            top_employees = top_employees.sort_values(
                ['monthly_score', employee_col], 
                ascending=[False, True]
            )
            
            for _, row in top_employees.iterrows():
                results.append({
                    'year_month': ym,
                    employee_col: row[employee_col],
                    'monthly_score': row['monthly_score'],
                    'message_count': row.get('message_count', 'N/A'),
                    'positive_count': row.get('positive_count', 'N/A'),
                    'negative_count': row.get('negative_count', 'N/A'),
                    'neutral_count': row.get('neutral_count', 'N/A')
                })
        
        return pd.DataFrame(results)
    
    def get_top_positive_employees(self, top_n=3, year_month=None):
        """
        Get top N positive employees
        
        Parameters:
        -----------
        top_n : int, default=3
            Number of top employees to return
        year_month : str or Period, optional
            Specific month to filter by
            
        Returns:
        --------
        pd.DataFrame : DataFrame with top positive employees
        """
        return self.get_top_employees(top_n=top_n, positive=True, year_month=year_month)
    
    def get_top_negative_employees(self, top_n=3, year_month=None):
        """
        Get top N negative employees
        
        Parameters:
        -----------
        top_n : int, default=3
            Number of top employees to return
        year_month : str or Period, optional
            Specific month to filter by
            
        Returns:
        --------
        pd.DataFrame : DataFrame with top negative employees
        """
        return self.get_top_employees(top_n=top_n, positive=False, year_month=year_month)
    
    def get_all_rankings(self, top_n=3):
        """
        Get all rankings (both positive and negative) for all months
        
        Parameters:
        -----------
        top_n : int, default=3
            Number of top employees to return per category
            
        Returns:
        --------
        dict : Dictionary with 'positive' and 'negative' DataFrames
        """
        return {
            'positive': self.get_top_positive_employees(top_n=top_n),
            'negative': self.get_top_negative_employees(top_n=top_n)
        }
    
    def get_overall_rankings(self, top_n=3):
        """
        Get overall rankings across all months (sum of all monthly scores)
        
        Parameters:
        -----------
        top_n : int, default=3
            Number of top employees to return per category
            
        Returns:
        --------
        dict : Dictionary with 'positive' and 'negative' DataFrames
        """
        employee_col = self.monthly_scores.columns[0]
        
        # Aggregate scores across all months
        overall_scores = self.monthly_scores.groupby(employee_col)['monthly_score'].sum().reset_index()
        overall_scores = overall_scores.sort_values('monthly_score', ascending=False)
        
        # Get top positive
        top_positive = overall_scores.nlargest(top_n, 'monthly_score').copy()
        top_positive = top_positive.sort_values(['monthly_score', employee_col], ascending=[False, True])
        
        # Get top negative
        top_negative = overall_scores.nsmallest(top_n, 'monthly_score').copy()
        top_negative = top_negative.sort_values(['monthly_score', employee_col], ascending=[True, True])
        
        return {
            'positive': top_positive,
            'negative': top_negative
        }

