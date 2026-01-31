"""
Sentiment Labeling Module
This module handles the automatic labeling of employee messages with sentiment categories.
"""

import pandas as pd
import numpy as np
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')


class SentimentLabeler:
    """
    A class to label text messages with sentiment (Positive, Negative, Neutral)
    """
    
    def __init__(self, method='vader', use_gpu=False):
        """
        Initialize the sentiment labeler
        
        Parameters:
        -----------
        method : str, default='vader'
            Method to use for sentiment analysis. Options: 'vader', 'transformer'
        use_gpu : bool, default=False
            Whether to use GPU for transformer models
        """
        self.method = method
        self.analyzer = None
        self.transformer_pipeline = None
        
        if method == 'vader':
            self.analyzer = SentimentIntensityAnalyzer()
        elif method == 'transformer':
            try:
                device = 0 if use_gpu else -1
                self.transformer_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=device
                )
            except Exception as e:
                print(f"Warning: Could not load transformer model. Falling back to VADER. Error: {e}")
                self.method = 'vader'
                self.analyzer = SentimentIntensityAnalyzer()
    
    def label_sentiment_vader(self, text):
        """
        Label sentiment using VADER sentiment analyzer
        
        Parameters:
        -----------
        text : str
            Text message to analyze
            
        Returns:
        --------
        str : 'Positive', 'Negative', or 'Neutral'
        """
        if pd.isna(text) or text == '':
            return 'Neutral'
        
        scores = self.analyzer.polarity_scores(str(text))
        compound = scores['compound']
        
        if compound >= 0.05:
            return 'Positive'
        elif compound <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    
    def label_sentiment_transformer(self, text):
        """
        Label sentiment using transformer model
        
        Parameters:
        -----------
        text : str
            Text message to analyze
            
        Returns:
        --------
        str : 'Positive', 'Negative', or 'Neutral'
        """
        if pd.isna(text) or text == '':
            return 'Neutral'
        
        try:
            # Truncate text if too long (transformer models have token limits)
            max_length = 512
            text_str = str(text)[:max_length]
            
            result = self.transformer_pipeline(text_str)[0]
            label = result['label']
            score = result['score']
            
            # Map transformer labels to our categories
            if 'POSITIVE' in label.upper() or 'POS' in label.upper():
                return 'Positive'
            elif 'NEGATIVE' in label.upper() or 'NEG' in label.upper():
                return 'Negative'
            else:
                return 'Neutral'
        except Exception as e:
            print(f"Error in transformer labeling: {e}. Falling back to Neutral.")
            return 'Neutral'
    
    def label_text(self, text):
        """
        Label a single text with sentiment
        
        Parameters:
        -----------
        text : str
            Text message to analyze
            
        Returns:
        --------
        str : 'Positive', 'Negative', or 'Neutral'
        """
        if self.method == 'vader':
            return self.label_sentiment_vader(text)
        elif self.method == 'transformer':
            return self.label_sentiment_transformer(text)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def label_dataframe(self, df, text_column='message'):
        """
        Label all messages in a dataframe
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing messages
        text_column : str, default='message'
            Name of the column containing text messages
            
        Returns:
        --------
        pd.DataFrame : DataFrame with added 'sentiment' column
        """
        df = df.copy()
        
        print(f"Labeling {len(df)} messages using {self.method} method...")
        
        if self.method == 'transformer' and self.transformer_pipeline is not None:
            # Process in batches for transformer models
            sentiments = []
            batch_size = 32
            for i in range(0, len(df), batch_size):
                batch = df[text_column].iloc[i:i+batch_size].tolist()
                batch_sentiments = [self.label_sentiment_transformer(text) for text in batch]
                sentiments.extend(batch_sentiments)
                if (i // batch_size + 1) % 10 == 0:
                    print(f"Processed {i + len(batch)}/{len(df)} messages...")
        else:
            sentiments = df[text_column].apply(self.label_text)
        
        df['sentiment'] = sentiments
        
        print(f"Sentiment labeling completed!")
        print(f"Sentiment distribution:")
        print(df['sentiment'].value_counts())
        
        return df

