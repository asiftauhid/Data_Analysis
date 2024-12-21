# RateMyProfessor Analysis

## Project Description
This project analyzes RateMyProfessor data to explore various aspects of professor ratings, including gender bias, tag correlations, and prediction models for key attributes like average ratings, difficulty, and "Pepper" awards. The analysis incorporates advanced statistical methods and machine learning models to derive meaningful insights.

## Group Members
- Asif Tauhid
- Haojie Cai
- Xiaokan Tian

## Data Preprocessing
1. **Data Loading**:
   - Merged numerical, qualitative, and tags datasets into a single dataframe.
   - Ensured proper labeling for each column.

2. **Missing Data Handling**:
   - Dropped columns with no ratings (e.g., `Number of Ratings = NaN`).
   - Filtered rows with `Proportion Retaking Class = NaN` for specific tasks.

3. **Tags Normalization**:
   - Normalized tag counts by dividing each tag by the number of ratings per professor.
   - Standardized tags with a mean of 0 and a standard deviation of 1.

4. **Filtering**:
   - Removed professors with fewer than 5 ratings to reduce noise and outliers.

## Questions Addressed
### Question 1: Gender Bias in Average Ratings
- **Methodology**:
  - Conducted Welch t-tests to compare average ratings.
  - Controlled for teaching experience by grouping professors into categories (new, experienced, senior).
- **Findings**:
  - Male professors have significantly higher ratings across all experience levels.

### Question 2: Gender Difference in Ratings Variance
- **Methodology**:
  - Performed Levene’s test to compare variances.
- **Findings**:
  - Female professors show higher variance in average ratings compared to male professors.

### Question 3: Effect Size of Gender Bias
- **Methodology**:
  - Calculated Cohen's d and bootstrapped for confidence intervals.
- **Findings**:
  - Small effect size for gender bias in ratings and variance, indicating minimal real-world impact.

### Question 4: Gender Difference in Tags
- **Methodology**:
  - Welch t-tests for each tag.
- **Findings**:
  - Significant gender differences in most tags, with "Hilarious," "Amazing Lectures," and "Caring" being the most gendered.

### Question 5: Gender Difference in Average Difficulty
- **Methodology**:
  - Conducted Welch t-tests and Mann-Whitney U tests.
- **Findings**:
  - No significant gender differences in difficulty ratings.

### Question 6: Confidence Interval for Difficulty Ratings
- **Methodology**:
  - Bootstrapped for 95% confidence intervals.
- **Findings**:
  - Confidence interval includes zero, supporting no significant gender difference.

### Question 7: Predicting Average Ratings
- **Methodology**:
  - Compared Linear, Ridge, and Lasso regression models.
- **Findings**:
  - Ridge regression performed best with R² = 0.8014 and RMSE = 0.1353.

### Question 8: Predicting Average Difficulty
- **Methodology**:
  - Used Ridge and Lasso regression models.
- **Findings**:
  - Ridge regression performed best with R² = 0.7267 and RMSE = 0.2541.

### Question 9: Predicting Tough Grader Tag
- **Methodology**:
  - Used Ridge and Lasso regression models.
- **Findings**:
  - Ridge regression performed better with R² = 0.5600 and RMSE = 0.3011.

### Question 10: Predicting Pepper Award
- **Methodology**:
  - Trained a Random Forest Classifier with SMOTE for class imbalance.
- **Findings**:
  - Model achieved AUROC = 0.81, with balanced precision, recall, and F1-scores.

## Extra Credit
- Analyzed regional differences in Pepper awards and correlations between tags and Pepper awards.
- Found that "Caring," "Good Feedback," and "Respected" tags strongly predict Pepper awards.

## Key Results
- Male professors consistently receive higher ratings and lower variance compared to female professors.
- Gender bias in tags and ratings exists but with small real-world impact.
- Predictive models like Ridge regression and Random Forest perform well in estimating ratings and Pepper awards.

## How to Run the Project
1. Download this directory
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the analysis scripts:
   ```bash
   python group_05_Capstone_project.py
   ```
