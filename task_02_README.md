# PRODIGY_DS_02: Data Cleaning & Exploratory Data Analysis

## Overview

This project performs data cleaning and exploratory data analysis on the Titanic dataset to identify patterns in survival rates and passenger characteristics.

## Dataset

**Titanic Dataset** - Kaggle competition data
- 891 passengers
- Features: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Fare, Embarked

## Requirements

- Python 3.x
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.2
- seaborn >= 0.11.1

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python task_02_eda.py
```

## Process

1. **Load Data** - Load Titanic dataset from local or online source
2. **Assess Quality** - Check for missing values, duplicates, data types
3. **Clean Data** - Handle missing values, remove duplicates, manage outliers
4. **Analyze** - Generate visualizations and statistical analysis
5. **Insights** - Extract key findings about survival patterns

## Visualizations Generated

- Survival Distribution
- Gender Distribution
- Age Distribution
- Passenger Class Distribution
- Fare Distribution
- Survival by Gender

## Key Findings

- Overall survival rate and breakdown by demographics
- Correlation between variables
- Distribution patterns in passenger data
- Survival rates by gender and passenger class

## Output

- task_02_eda.png - 6-panel visualization (300 DPI)
- Console statistics and insights

## Author

Saransh Lives - Prodigy Infotech Data Science Internship
