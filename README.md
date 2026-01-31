# Employee Sentiment Analysis Project

## Project Overview

This project implements a comprehensive sentiment analysis system for employee messages. The system analyzes unlabeled employee messages to assess sentiment and engagement, providing insights into employee satisfaction and identifying potential flight risks.

## Project Structure

```
employee-sentiment-analysis/
│
├── data/
│   └── test.csv                    # Input dataset (unlabeled employee messages)
│
├── notebooks/
│   └── employee_sentiment_analysis.ipynb  # Main analysis notebook
│
├── src/
│   ├── __init__.py
│   ├── sentiment_labeling.py       # Sentiment labeling module
│   ├── eda.py                      # Exploratory data analysis module
│   ├── scoring.py                  # Employee score calculation module
│   ├── ranking.py                  # Employee ranking module
│   ├── flight_risk.py              # Flight risk identification module
│   └── modeling.py                 # Predictive modeling module
│
├── visualizations/
│   ├── sentiment_distribution.png
│   ├── sentiment_trends.png
│   ├── employee_activity.png
│   ├── message_length_analysis.png
│   ├── employee_rankings.png
│   └── model_performance.png
│
├── reports/
│   └── final_report.docx           # Comprehensive project report
│
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
└── .gitignore
```

## Features

### 1. Sentiment Labeling
- Automatically labels each employee message as **Positive**, **Negative**, or **Neutral**
- Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analyzer
- Supports transformer-based models for enhanced accuracy (optional)

### 2. Exploratory Data Analysis (EDA)
- Comprehensive data structure analysis
- Sentiment distribution visualization
- Temporal trend analysis
- Employee activity patterns
- Message length and word count analysis

### 3. Employee Score Calculation
- Computes monthly sentiment scores for each employee
- Scoring system:
  - Positive Message: +1
  - Negative Message: -1
  - Neutral Message: 0
- Scores reset at the beginning of each month

### 4. Employee Ranking
- Identifies top 3 positive employees per month
- Identifies top 3 negative employees per month
- Rankings sorted by score (descending) then alphabetically

### 5. Flight Risk Identification
- Identifies employees at risk of leaving
- Criteria: 4+ negative messages in a rolling 30-day period
- Provides detailed risk analysis and summary statistics

### 6. Predictive Modeling
- Linear regression model to predict sentiment scores
- Features include:
  - Message frequency
  - Message length (average and total)
  - Word count (average and total)
  - Sentiment ratios (positive/negative/neutral)
  - Temporal features (month)
- Model evaluation using MSE, RMSE, MAE, and R² metrics

## Installation

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure the dataset `test.csv` is in the `data/` directory

## Usage

### Running the Analysis

1. Open the Jupyter notebook:
```bash
jupyter notebook notebooks/employee_sentiment_analysis.ipynb
```

2. Run all cells sequentially to perform the complete analysis

3. The notebook will:
   - Load and preprocess the data
   - Label sentiments
   - Perform EDA
   - Calculate employee scores
   - Generate rankings
   - Identify flight risks
   - Build and evaluate the predictive model

### Output Files

After running the notebook, you'll find:
- **Processed data**: `data/processed_data.csv` (with sentiment labels)
- **Monthly scores**: `data/monthly_scores.csv`
- **Flight risks**: `data/flight_risks.csv` (if any identified)
- **Visualizations**: All charts saved in `visualizations/` folder

## Key Findings Summary

### Top 3 Positive Employees
*(Results will vary based on your dataset)*

### Top 3 Negative Employees
*(Results will vary based on your dataset)*

### Flight Risk Employees
*(Results will vary based on your dataset)*

## Technical Details

### Sentiment Analysis Method
- **Primary**: VADER Sentiment Analyzer
  - Fast and efficient
  - Works well with informal text
  - No training data required
- **Alternative**: Transformer-based models (RoBERTa)
  - Higher accuracy potential
  - Requires GPU for optimal performance
  - Can be enabled by changing `method='transformer'` in the notebook

### Model Performance
The linear regression model is evaluated using:
- **Mean Squared Error (MSE)**: Measures average squared differences
- **Root Mean Squared Error (RMSE)**: Standard deviation of residuals
- **Mean Absolute Error (MAE)**: Average absolute differences
- **R² Score**: Proportion of variance explained

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computing
- matplotlib: Plotting and visualization
- seaborn: Statistical data visualization
- scikit-learn: Machine learning (linear regression)
- torch: PyTorch (for transformer models, optional)
- transformers: Hugging Face transformers (optional)
- vaderSentiment: VADER sentiment analyzer
- jupyter: Jupyter notebook environment

See `requirements.txt` for complete list with versions.

## Project Deliverables

1. **Code Submission**: 
   - Main notebook: `notebooks/employee_sentiment_analysis.ipynb`
   - Supporting modules: `src/` directory
   - Well-commented and documented code

2. **Final Report**: 
   - Comprehensive document in `reports/final_report.docx`
   - Includes methodology, findings, and conclusions

3. **Visualizations**: 
   - All charts saved in `visualizations/` folder
   - High-resolution PNG files

4. **README**: 
   - This file with project summary and key findings

## Notes

- The project is designed to be reproducible
- All code is well-documented with comments and explanations
- The analysis can be easily adapted for different datasets
- Flight risk threshold (4 negative messages) can be adjusted if needed

## Contact

For questions or issues, please refer to the project documentation or contact the project maintainer.

---

**Note**: This is an internal evaluation project. Please do not share project information publicly.

