
# Movie Data Analysis Replication

This project analyzes a dataset of movie ratings to explore statistical questions about movie ratings, demographics, and viewing preferences. The project actually replicates some part of a research paper([click here to view](https://blog.pascallisch.net/wp-content/uploads/2017/11/proj110107.pdf)) written by Pascal Wallisch and his cowriter Jake Alden Whritner. Here, various statistical tests were utilized to provide insights into viewer behavior and preferences.

## Team Members
- Asif Tauhid
- Xiaokan Tian
- Haojie Cai

## Research Questions and Key Findings

1. **Are movies that are more popular rated higher than less popular movies?**
   - **Methodology**: Used Welch's t-test to compare average ratings of popular and unpopular movies (split by median rating count).
   - **Findings**: Popular movies have significantly higher average ratings (2.8683) compared to unpopular movies (2.4009).

2. **Are newer movies rated differently than older movies?**
   - **Methodology**: Used an independent t-test to compare average ratings of newer and older movies (split by median release year).
   - **Findings**: No significant difference in ratings between new and old movies.

3. **Is the enjoyment of 'Shrek (2001)' gendered?**
   - **Methodology**: Conducted Mann-Whitney U and Kolmogorov-Smirnov (KS) tests.
   - **Findings**: No significant difference in ratings between male and female viewers.

4. **What proportion of movies are rated differently by male and female viewers?**
   - **Methodology**: Applied Mann-Whitney U test for each movie.
   - **Findings**: Approximately 12.5% of movies show gender-based rating differences.

5. **Do only children enjoy 'The Lion King (1994)' more than people with siblings?**
   - **Methodology**: Used Mann-Whitney U test to compare ratings.
   - **Findings**: No significant difference in enjoyment between only children and those with siblings.

6. **What proportion of movies exhibit an 'only child effect'?**
   - **Methodology**: Mann-Whitney U tests across all movies.
   - **Findings**: Only 1.75% of movies show significant differences in ratings between only children and those with siblings.

7. **Do social watchers enjoy 'The Wolf of Wall Street (2013)' more than solo watchers?**
   - **Methodology**: Conducted Mann-Whitney U test and power analysis.
   - **Findings**: No significant difference in ratings between the two groups.

8. **What proportion of movies exhibit a 'social watching effect'?**
   - **Methodology**: Mann-Whitney U tests across 400 movies.
   - **Findings**: Approximately 2.5% of movies show significant differences based on watching preferences.

9. **Is the ratings distribution of 'Home Alone (1990)' different from 'Finding Nemo (2003)'?**
   - **Methodology**: Kolmogorov-Smirnov test and power analysis.
   - **Findings**: Significant difference in rating distributions between the two movies.

10. **Are movie franchises of consistent quality?**
    - **Methodology**: Used Mann-Whitney U and Kruskal-Wallis H tests.
    - **Findings**: 87.5% of franchises analyzed exhibit inconsistent quality.

## Dataset

- **Source**: `movieReplicationSet.csv`
- **Size**: Contains movie ratings and metadata, including demographic and preference information.

## How to Run

1. Download this directory
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the analysis script:
   ```bash
   python movieDataAnalysisReplication.py
   ```
   
## Technologies Used
- Python (Pandas, NumPy, SciPy, Matplotlib)
- Statistical Tests: Welch's t-test, Mann-Whitney U test, Kolmogorov-Smirnov test, Kruskal-Wallis H test.

## Results and Discussions

The results demonstrate diverse patterns in movie ratings based on popularity, gender, social watching preferences, and demographic factors. Key findings highlight significant and non-significant trends, suggesting areas for further research and potential improvement in personalized movie recommendations.


