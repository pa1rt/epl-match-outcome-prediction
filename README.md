# English Premier League Match Outcome Prediction

## DS-670: Capstone — Big Data & Data Science

### Project Overview
This project explores the use of machine learning to predict English Premier League soccer match outcomes using historical team-performance data.

The prediction task is a three-class classification problem:
- Home Win (H)
- Draw (D)
- Away Win (A)

### Dataset
- Source: Football-Data.co.uk
- Seasons: 2021–22 through 2025–26
- Total matches: 1,900
- Data includes match results, goals, and team statistics.

### Methodology
1. Collect and clean historical match data.
2. Perform exploratory data analysis.
3. Engineer features using previous matches.
4. Split training and testing data chronologically.
5. Train and compare classification models.
6. Evaluate results using accuracy, macro F1-score, and confusion matrices.

### Preliminary Results

| Model | Accuracy |
|---|---|
| Majority-Class Baseline | 42.4% |
| Logistic Regression | 46.7% |
| Random Forest | 45.6% |

Logistic Regression achieved the highest preliminary accuracy. Predicting draws remains a significant challenge.

### Technologies
- Python
- pandas
- scikit-learn
- Matplotlib

### Future Work
- Improve historical team-performance features.
- Explore additional classification models.
- Improve prediction performance for draws.
- Evaluate model generalization and limitations.

### Project Status
In progress 
