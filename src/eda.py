"""
Exploratory Data Analysis Module
This module handles data exploration and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class EDAAnalyzer:
    """
    A class to perform exploratory data analysis on employee sentiment data
    """
    
    def __init__(self, df):
        """
        Initialize EDA analyzer
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with employee messages and sentiment labels
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
        
        # Extract temporal features
        if 'date' in self.df.columns:
            self.df['year'] = self.df['date'].dt.year
            self.df['month'] = self.df['date'].dt.month
            self.df['year_month'] = self.df['date'].dt.to_period('M')
            self.df['day_of_week'] = self.df['date'].dt.day_name()
            self.df['hour'] = self.df['date'].dt.hour
    
    def basic_info(self):
        """
        Display basic information about the dataset
        
        Returns:
        --------
        dict : Dictionary with basic statistics
        """
        info = {
            'total_records': len(self.df),
            'columns': list(self.df.columns),
            'data_types': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicate_records': self.df.duplicated().sum()
        }
        
        print("=" * 60)
        print("BASIC DATASET INFORMATION")
        print("=" * 60)
        print(f"Total Records: {info['total_records']}")
        print(f"\nColumns ({len(self.df.columns)}):")
        for col in self.df.columns:
            print(f"  - {col}")
        print(f"\nMissing Values:")
        for col, count in info['missing_values'].items():
            if count > 0:
                print(f"  - {col}: {count} ({count/len(self.df)*100:.2f}%)")
        print(f"\nDuplicate Records: {info['duplicate_records']}")
        print("=" * 60)
        
        return info
    
    def sentiment_distribution(self, save_path=None):
        """
        Analyze and visualize sentiment distribution
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the visualization
        """
        if 'sentiment' not in self.df.columns:
            print("Error: 'sentiment' column not found in dataframe")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        sentiment_counts = self.df['sentiment'].value_counts()
        axes[0].bar(sentiment_counts.index, sentiment_counts.values, 
                   color=['#2ecc71', '#e74c3c', '#95a5a6'])
        axes[0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Sentiment', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(sentiment_counts.values):
            axes[0].text(i, v + max(sentiment_counts.values) * 0.01, 
                        str(v), ha='center', fontweight='bold')
        
        # Pie chart
        colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}
        pie_colors = [colors.get(s, '#95a5a6') for s in sentiment_counts.index]
        axes[1].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                   autopct='%1.1f%%', startangle=90, colors=pie_colors)
        axes[1].set_title('Sentiment Proportion', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
        
        return sentiment_counts
    
    def temporal_trends(self, save_path=None):
        """
        Analyze sentiment trends over time
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the visualization
        """
        if 'date' not in self.df.columns or 'sentiment' not in self.df.columns:
            print("Error: Required columns ('date', 'sentiment') not found")
            return
        
        # Group by date
        daily_sentiment = self.df.groupby([self.df['date'].dt.date, 'sentiment']).size().unstack(fill_value=0)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Line plot
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            if sentiment in daily_sentiment.columns:
                axes[0].plot(daily_sentiment.index, daily_sentiment[sentiment], 
                           label=sentiment, marker='o', markersize=3)
        
        axes[0].set_title('Daily Sentiment Trends', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Date', fontsize=12)
        axes[0].set_ylabel('Number of Messages', fontsize=12)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Monthly aggregation
        if 'year_month' in self.df.columns:
            monthly_sentiment = self.df.groupby(['year_month', 'sentiment']).size().unstack(fill_value=0)
            monthly_sentiment.plot(kind='bar', ax=axes[1], width=0.8)
            axes[1].set_title('Monthly Sentiment Distribution', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Month', fontsize=12)
            axes[1].set_ylabel('Number of Messages', fontsize=12)
            axes[1].legend(title='Sentiment')
            axes[1].grid(axis='y', alpha=0.3)
            axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def employee_activity(self, top_n=10, save_path=None):
        """
        Analyze employee activity patterns
        
        Parameters:
        -----------
        top_n : int, default=10
            Number of top employees to display
        save_path : str, optional
            Path to save the visualization
        """
        if 'employee' not in self.df.columns and 'employee_id' not in self.df.columns:
            print("Error: Employee column not found")
            return
        
        employee_col = 'employee' if 'employee' in self.df.columns else 'employee_id'
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Total messages per employee
        employee_counts = self.df[employee_col].value_counts().head(top_n)
        axes[0].barh(range(len(employee_counts)), employee_counts.values)
        axes[0].set_yticks(range(len(employee_counts)))
        axes[0].set_yticklabels(employee_counts.index)
        axes[0].set_title(f'Top {top_n} Most Active Employees', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Number of Messages', fontsize=12)
        axes[0].grid(axis='x', alpha=0.3)
        
        # Sentiment distribution by employee (top employees)
        top_employees = employee_counts.index[:top_n]
        employee_sentiment = self.df[self.df[employee_col].isin(top_employees)].groupby(
            [employee_col, 'sentiment']).size().unstack(fill_value=0)
        employee_sentiment.plot(kind='barh', ax=axes[1], stacked=True, 
                              color=['#2ecc71', '#e74c3c', '#95a5a6'])
        axes[1].set_title(f'Sentiment Distribution - Top {top_n} Employees', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Number of Messages', fontsize=12)
        axes[1].legend(title='Sentiment')
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def message_length_analysis(self, save_path=None):
        """
        Analyze message length patterns
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the visualization
        """
        # Find message column
        message_col = None
        for col in ['message', 'text', 'content', 'body']:
            if col in self.df.columns:
                message_col = col
                break
        
        if message_col is None:
            print("Error: Message column not found")
            return
        
        self.df['message_length'] = self.df[message_col].astype(str).str.len()
        self.df['word_count'] = self.df[message_col].astype(str).str.split().str.len()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Message length distribution
        axes[0, 0].hist(self.df['message_length'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Message Length Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Character Count', fontsize=10)
        axes[0, 0].set_ylabel('Frequency', fontsize=10)
        axes[0, 0].grid(alpha=0.3)
        
        # Word count distribution
        axes[0, 1].hist(self.df['word_count'], bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].set_title('Word Count Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Word Count', fontsize=10)
        axes[0, 1].set_ylabel('Frequency', fontsize=10)
        axes[0, 1].grid(alpha=0.3)
        
        # Message length by sentiment
        if 'sentiment' in self.df.columns:
            sentiment_groups = self.df.groupby('sentiment')['message_length']
            axes[1, 0].boxplot([sentiment_groups.get_group(s).values 
                               for s in sentiment_groups.groups.keys()],
                              labels=sentiment_groups.groups.keys())
            axes[1, 0].set_title('Message Length by Sentiment', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('Character Count', fontsize=10)
            axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Word count by sentiment
        if 'sentiment' in self.df.columns:
            sentiment_groups = self.df.groupby('sentiment')['word_count']
            axes[1, 1].boxplot([sentiment_groups.get_group(s).values 
                               for s in sentiment_groups.groups.keys()],
                              labels=sentiment_groups.groups.keys())
            axes[1, 1].set_title('Word Count by Sentiment', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('Word Count', fontsize=10)
            axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def generate_summary_report(self):
        """
        Generate a comprehensive summary report
        
        Returns:
        --------
        dict : Dictionary with summary statistics
        """
        summary = {}
        
        # Basic stats
        summary['total_messages'] = len(self.df)
        summary['unique_employees'] = self.df.get('employee', 
            self.df.get('employee_id', pd.Series([0]))).nunique()
        
        # Sentiment stats
        if 'sentiment' in self.df.columns:
            sentiment_counts = self.df['sentiment'].value_counts()
            summary['sentiment_distribution'] = sentiment_counts.to_dict()
            summary['positive_percentage'] = (sentiment_counts.get('Positive', 0) / len(self.df)) * 100
            summary['negative_percentage'] = (sentiment_counts.get('Negative', 0) / len(self.df)) * 100
            summary['neutral_percentage'] = (sentiment_counts.get('Neutral', 0) / len(self.df)) * 100
        
        # Temporal stats
        if 'date' in self.df.columns:
            summary['date_range'] = {
                'start': str(self.df['date'].min()),
                'end': str(self.df['date'].max()),
                'span_days': (self.df['date'].max() - self.df['date'].min()).days
            }
        
        return summary

