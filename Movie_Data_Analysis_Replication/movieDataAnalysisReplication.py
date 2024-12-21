# Group 16 Midterm
# Group Members:
# Asif Tauhid
# Xiaokan Tian
# Haojie Cai

########################################################################################################################

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('E:/2024_fall/1001/midterm/movieReplicationSet.csv')

def check_var_similar(df_1, df_2): 
    levene_stat, levene_p = stats.levene(df_1, df_2, center='mean')
    if levene_p > 0.05: 
        print("variances are similar")
    else: 
        print("variances are not similar")


def cohens_d(df1, df2):
    pooled_std = np.sqrt((np.std(df1, ddof=1)**2 + np.std(df2, ddof=1)**2) / 2)
    d = abs((np.mean(df1) - np.mean(df2))) / pooled_std
    return d

########################################################################################################################

# Q1
ratings_data = df.iloc[:, :400]
ratings_count = ratings_data.notna().sum()

# split high low popularity
median_count = ratings_count.median()

# get movie rating means
movie_mean_rating = ratings_data.mean()

# get indexes
high_popularity_movies_idx = ratings_count[ratings_count > median_count].index
low_popularity_movies_idx = ratings_count[ratings_count <= median_count].index

high_pop_movie_df = movie_mean_rating[high_popularity_movies_idx]
low_pop_movie_df = movie_mean_rating[low_popularity_movies_idx]

# Check Distribution
plt.figure(figsize=(10, 5))
plt.hist(movie_mean_rating, bins=30)
plt.title('Distribution of Rating Mean per Movie')
plt.xlabel('Mean of Ratings')
plt.ylabel('Frequency')
plt.show()


# test if variances are similar 
check_var_similar(high_pop_movie_df, low_pop_movie_df)
# variances not similar, use Welch t-test

# Welch t-test
t_stat, p_value = stats.ttest_ind(
    high_pop_movie_df, 
    low_pop_movie_df, 
    equal_var=False, 
    nan_policy='omit'
)

alpha = 0.005

is_significant = p_value < alpha
# check values
t_stat, p_value, is_significant

# compare high low pop movies rating
high_pop_movie_mean = high_pop_movie_df.mean()
low_pop_movie_mean = low_pop_movie_df.mean()
high_pop_movie_mean, low_pop_movie_mean

########################################################################################################################

# Q2
movie_titles = ratings_data.columns
# get release year
release_years = movie_titles.str.extract(r'\((\d{4})\)').astype(float)
median_year = release_years.median().values[0]
print(median_year)

# get movie indexes
new_movies_idx = movie_titles[release_years[0] > median_year]
old_movies_idx = movie_titles[release_years[0] <= median_year]

new_movies_df = movie_mean_rating[new_movies_idx]
old_movies_df = movie_mean_rating[old_movies_idx]

# check distribution
plt.figure(figsize=(8, 6))
sns.violinplot(x=['New'] * len(new_movies_df) + ['Old'] * len(old_movies_df),
               y=pd.concat([new_movies_df, old_movies_df]),
               palette='Set2')
plt.title("Distribution of Mean Ratings by New and Old Movies")
plt.ylabel("Mean Rating")
plt.show()

# test if variances are similar 
check_var_similar(new_movies_df, old_movies_df)
# variances similar, use independent sample t-test

# independent sample t-test
t_stat, p_value = stats.ttest_ind(
    new_movies_df, 
    old_movies_df, 
    equal_var=True, 
    nan_policy='omit'
)

alpha = 0.005

# check values
is_significant = p_value < alpha
t_stat, p_value, is_significant

########################################################################################################################

#Q3
shrek_ratings = df['Shrek (2001)']
gender = df.iloc[:, 474]

male_ratings = shrek_ratings[gender == 2].dropna()
female_ratings = shrek_ratings[gender == 1].dropna() 

# check distribution
plt.figure(figsize=(12, 6))
plt.hist(male_ratings, bins=20, alpha=0.7, label='Male', color='blue', edgecolor='black')
plt.hist(female_ratings, bins=20, alpha=0.7, label='Female', color='orange', edgecolor='black')
plt.title('Distribution of Mean Ratings for Shrek')
plt.xlabel('Mean Rating')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# U Test
U_stat, p_value = stats.mannwhitneyu(male_ratings, female_ratings, alternative='two-sided')
alpha = 0.005
is_significant = p_value < alpha
U_stat, p_value, is_significant

# KS Test
ks_stat, p_value = stats.ks_2samp(male_ratings, female_ratings)
alpha = 0.005
is_significant = p_value < alpha
ks_stat, p_value, is_significant

########################################################################################################################

# Q4 What proportion of movies are rated differently by male and female viewers?

# Extract movie ratings and gender column
ratings_data = df.iloc[:, :400]
gender_data = df.iloc[:, 474]

# Separate ratings by gender
male_ratings_dict = {}
female_ratings_dict = {}

# Filter movies in terms of gender
for movie in ratings_data.columns:
    movie_ratings = ratings_data[movie]

    male_ratings_dict[movie] = movie_ratings[gender_data == 2].dropna()
    female_ratings_dict[movie] = movie_ratings[gender_data == 1].dropna()

#print(male_ratings_dict)
#print(female_ratings_dict)


# Calculate mean ratings for each movie by gender
male_means = {movie: ratings.mean() for movie, ratings in male_ratings_dict.items()}
female_means = {movie: ratings.mean() for movie, ratings in female_ratings_dict.items()}

# Create a DataFrame for plotting
gender_ratings_df = pd.DataFrame({
    'Movie': list(male_means.keys()),
    'Male Ratings': list(male_means.values()),
    'Female Ratings': list(female_means.values())
})
gender_ratings_df['Rating Difference'] = gender_ratings_df['Male Ratings'] - gender_ratings_df['Female Ratings']

# Plot: Distribution of Rating Differences Between Male and Female Viewers
plt.figure()
sns.histplot(gender_ratings_df['Rating Difference'])
plt.xlabel('Difference in Ratings')
plt.ylabel('Number of Movies')
plt.title("Distribution of Rating Differences Between Male and Female Viewers for Each Movie")
plt.show()


# Set significance level
alpha = 0.005
significant_movies_count = 0
total_movies = ratings_data.shape[1]


# Mannwhitney U test for each movie
for movie in ratings_data.columns:
    male_ratings = male_ratings_dict[movie]
    female_ratings = female_ratings_dict[movie]
    
    #print(len(male_ratings))
    #print(len(female_ratings))
    
    # Only perform ratings if both genders exist
    if not male_ratings.empty and not female_ratings.empty:
        stat, p_value = stats.mannwhitneyu(male_ratings, female_ratings, alternative='two-sided')
        
        # Check if result is significant
        if p_value < alpha:
            significant_movies_count += 1
            
            
print(f"Number of significant movies: {significant_movies_count}")
print(f"Number of total movies: {total_movies}")


# Gender based rating difference
proportion = significant_movies_count / total_movies
print(f"The proportion of movies rated differently by male and female viewers is: {proportion:.4f}")

########################################################################################################################

# Q5 Do people who are only children enjoy ‘The Lion King(1994)’ more than people with siblings?

# Get the ratings of 'The Lion King (1994)' by 'only-child' status
lion_king_ratings = df['The Lion King (1994)']
only_child_data = df.iloc[:, 475]

# Separate ratings for only child and with siblings
only_child_ratings = lion_king_ratings[only_child_data == 1].dropna()
sibling_ratings = lion_king_ratings[only_child_data == 0].dropna()

# print(only_child_ratings)
# print(sibling_ratings)


# Create a DataFrame for plotting
lion_king_df = pd.DataFrame({
    'Rating': pd.concat([only_child_ratings, sibling_ratings], ignore_index=True),
    'Status': ['Only Child'] * len(only_child_ratings) + ['With Siblings'] * len(sibling_ratings)
})

# Violin plot of ratings by only-child status
plt.figure()
sns.violinplot(data=lion_king_df, x='Status', y='Rating')
plt.xlabel('Status')
plt.ylabel('Rating')
plt.title("Ratings of 'The Lion King (1994)' by Only-Child Status")
plt.show()


# Run Mannwhitney Test
alpha = 0.005

if not only_child_ratings.empty and not sibling_ratings.empty:
    stat, p_value = stats.mannwhitneyu(only_child_ratings, sibling_ratings, alternative='two-sided')

if p_value < alpha:
    print(f"p-value: {p_value:.4f} < alpha. We reject H0")
else:
    print(f"p-value: {p_value:.4f} > alpha. We cannot reject H0)")

########################################################################################################################

# Q6

# Extract the only-child data column
only_child_data = df.iloc[:, 475]

# Dictionaries for each movies
only_child_ratings_dict = {}
sibling_ratings_dict = {}

# Filter ratings by only-child status for each movie
for movie in ratings_data.columns:
    movie_ratings = ratings_data[movie]
    
    only_child_ratings_dict[movie] = movie_ratings[only_child_data == 1].dropna()
    sibling_ratings_dict[movie] = movie_ratings[only_child_data == 0].dropna()

# print(only_child_ratings_dict)
# print(singling_ratings_dict)


# Calculate mean ratings for each movie by only-child status
only_child_means = {movie: ratings.mean() for movie, ratings in only_child_ratings_dict.items()}
sibling_means = {movie: ratings.mean() for movie, ratings in sibling_ratings_dict.items()}

# Create a DataFrame for plotting
child_sibling_df = pd.DataFrame({
    'Movie': list(only_child_means.keys()),
    'Only Child Ratings': list(only_child_means.values()),
    'Sibling Ratings': list(sibling_means.values())
})
child_sibling_df['Rating Difference'] = child_sibling_df['Only Child Ratings'] - child_sibling_df['Sibling Ratings']

# Plot: Distribution of Rating Differences Between Only Children and Siblings Across Movies
plt.figure()
sns.histplot(child_sibling_df['Rating Difference'])
plt.xlabel('Difference in Ratings (Only Child - Sibling)')
plt.ylabel('Number of Movies')
plt.title("Distribution of Rating Differences Between Only Child and Sibling Viewers for Each Movie")
plt.show()


# Applying Mannwhitney test

# Initializing the variables
alpha = 0.005
significant_movies_count = 0
total_movies = ratings_data.shape[1]

for movie in ratings_data.columns:
    only_child_ratings = only_child_ratings_dict[movie]
    sibling_ratings = sibling_ratings_dict[movie]
    
    # Perform test if there are ratings for both groups
    if not only_child_ratings.empty and not sibling_ratings.empty:
        stat, p_value = stats.mannwhitneyu(only_child_ratings, sibling_ratings, alternative='two-sided')
        
        # Check if result is significant
        if p_value < alpha:
            significant_movies_count += 1

print(f"Number of significant movies: {significant_movies_count}")
print(f"Number of total movies: {total_movies}")

#proportion

proportion = significant_movies_count / total_movies
print(f"Proportion of movies with a significant 'only child effect': {proportion:.4f}")

########################################################################################################################

# Q7

# socially enjoy movie and ratings of movie 'The Wolf of Wall Street (2013)'
alone_rating = df.iloc[:, 476]
wolf = df['The Wolf of Wall Street (2013)']


# divide into groups based on socially enjoyment and dropna
alone_1 = wolf[alone_rating == 1].dropna()
alone_0 = wolf[alone_rating == 0].dropna()
alone_0


# find sample size
print("The size of alone sample", len(alone_1))
print("The size of social sample", len(alone_0))

# test if variances are similar 
check_var_similar(alone_1, alone_0)


# Combine the data into a single DataFrame for plotting
plot_data = pd.DataFrame({
    'Mean_Rating': np.concatenate([alone_1, alone_0]),
    'Enjoy_Alone': ['Alone'] * len(alone_1) + ['Social'] * len(alone_0)
})

# Plot violin plots
sns.violinplot(x='Enjoy_Alone', y='Mean_Rating', data=plot_data)
plt.title('Distribution of Mean Ratings by Popularity Group')
plt.show()


# Welch t-test (large sample size, similar standard deviation, slightly skewed)
t_stat, p_value = stats.ttest_ind(
    alone_0,
    alone_1, 
    equal_var=False, 
    nan_policy='omit'
)

alpha = 0.005

is_significant = p_value < alpha
print("test statistic:", t_stat)
print("p-value:", p_value)
print("is significant:", is_significant)


# find power
from statsmodels.stats.power import TTestIndPower
d = cohens_d(alone_0, alone_1)
n = len(alone_0) + len(alone_1)
power = TTestIndPower().solve_power(effect_size=d, nobs1=n, alpha=alpha, alternative='two-sided')
print("power:", power)


# Mann-Whitney U test
u_stat, p_value = stats.mannwhitneyu(alone_0, alone_1)

is_significant = p_value < alpha
print("U statistic:", u_stat)
print("p-value:", p_value)
print("is significant:", is_significant)

# find power of U test

n_simulations = 10000  # Number of simulations to estimate power
alpha = 0.005  # Significance level
sample_size_0 = len(alone_0)  # Size of first sample
sample_size_1 = len(alone_1)  # Size of second sample

# Simulate samples
np.random.seed(42)  # For reproducibility
effect_size = cohens_d(alone_0,alone_1)  # Define the effect size (mean difference between the groups)
print("effect size:", effect_size)

# Estimate the power
significant_results = 0

for _ in range(n_simulations):
    # Generate two samples with a specified effect size
    simulated_sample_0 = np.random.normal(0, 1, sample_size_0)
    simulated_sample_1 = np.random.normal(effect_size, 1, sample_size_1)

    # Perform Mann-Whitney U test
    u_stat, p_value = stats.mannwhitneyu(simulated_sample_0, simulated_sample_1, alternative="two-sided")

    # Count if the result is significant
    if p_value < alpha:
        significant_results += 1

# Calculate power
power = significant_results / n_simulations
print("Estimated Power:", power)

########################################################################################################################

# Q8

# socially enjoy movie and ratings of movie 'The Wolf of Wall Street (2013)'
alone_rating = df.iloc[:, 476]
df_8 = df.iloc[:, :400]


# divide into groups based on socially enjoyment
alone_df = df_8[alone_rating == 1]
social_df = df_8[alone_rating == 0]


# parametric test: t-test
significant_num = 0
results = []

for i in range(400):
    # drop nan values
    alone_df_i = alone_df.iloc[:, i].dropna()
    social_df_i = social_df.iloc[:, i].dropna()

    # check sample size 
    alone_size = len(alone_df_i)
    social_size = len(social_df_i)

    # use Welch t-test
    t_stat, p_value = stats.ttest_ind(
    alone_df_i,
    social_df_i, 
    equal_var=False, 
    nan_policy='omit'
)

    is_significant = p_value < alpha

    results.append({
        't_stat': t_stat,
        'p_value': p_value,
        'is_significant': is_significant,
    })

    if (is_significant):
        significant_num += 1 


# visualize the distribution of p-values
p_values = [result['p_value'] for result in results]
plt.hist(p_values, bins=50)
plt.title('Distribution of p-values')
plt.xlabel('p-value')
plt.ylabel('count')
plt.show()

print("proportion of significant results:", significant_num / 400)

# non-parametric test: u-test
significant_num = 0
results = []

for i in range(400):
    # drop nan values
    alone_df_i = alone_df.iloc[:, i].dropna()
    social_df_i = social_df.iloc[:, i].dropna()

    # check sample size 
    alone_size = len(alone_df_i)
    social_size = len(social_df_i)

    # use Mann-Whitney U test since not sure about normality, variances, and sample size
    t_stat, p_value = stats.mannwhitneyu(
        alone_df_i, 
        social_df_i, 
        alternative='two-sided',
        nan_policy='omit'
    )

    is_significant = p_value < alpha

    results.append({
        't_stat': t_stat,
        'p_value': p_value,
        'is_significant': is_significant,
    })

    if (is_significant):
        significant_num += 1    
        print(alone_df_i.name)


# visualize the distribution of p-values
p_values = [result['p_value'] for result in results]
plt.hist(p_values, bins=200)
plt.axvline(x=0.005, color='red', linestyle='--', linewidth=1)
plt.title('Distribution of p-values')
plt.xlabel('p-value')
plt.ylabel('count')
plt.show()

print("proportion of significant results:", significant_num / 400)

########################################################################################################################

# Q9

# the ratings of movies
HA = df['Home Alone (1990)'].dropna()
FN = df['Finding Nemo (2003)'].dropna()


# visualize the distribution of ratings
# Combine the data into a single DataFrame for plotting
plot_data = pd.DataFrame({
    'Mean_Rating': np.concatenate([HA, FN]),
    'Movie': ['Home Alone (1990)'] * len(HA) + ['Finding Nemo (2003)'] * len(FN)
})

# Plot violin plots
sns.violinplot(x='Movie', y='Mean_Rating', data=plot_data)
plt.title('Distribution of Mean Ratings by Movies')
plt.show()


# ks test to test if the distributions are the same
ks_stat, ks_p = stats.ks_2samp(HA, FN)

is_significant = ks_p < alpha
print("test statistic:", ks_stat)
print("p-value:", ks_p)
print("is significant:", is_significant)


n_simulations = 10000  # Number of simulations to estimate power
alpha = 0.005  # Significance level
sample_size_0 = len(HA)  # Size of first sample
sample_size_1 = len(FN)  # Size of second sample
print("sample size:", sample_size_0 + sample_size_1)
# Simulate samples
np.random.seed(42)  # For reproducibility
effect_size = cohens_d(HA, FN)  # Define the effect size (mean difference between the groups)
print("effect size:", effect_size)

# Estimate the power
significant_results = 0

for _ in range(n_simulations):
    # Generate two samples with a specified effect size
    simulated_sample_0 = np.random.normal(0, 1, sample_size_0)
    simulated_sample_1 = np.random.normal(effect_size, 1, sample_size_1)

    # Perform Mann-Whitney U test
    u_stat, p_value = stats.ks_2samp(simulated_sample_0, simulated_sample_1)

    # Count if the result is significant
    if p_value < alpha:
        significant_results += 1

# Calculate power
power = significant_results / n_simulations
print("Estimated Power:", power)

########################################################################################################################

# Q10
# initializ

df_10 = df.iloc[:, :400]

franchises = [
    'Star Wars',
    'Harry Potter',
    'The Matrix',
    'Indiana Jones',
    'Jurassic Park',
    'Pirates of the Caribbean',
    'Toy Story',
    'Batman'
]

movie_titlles = df_10.columns


# parametric tests
significant_num = 0
significant_movies = []

for franchise in franchises:
    # find the movies in the franchise
    series = [title for title in movie_titlles if franchise in title]

    # drop series with less than 2 movies
    if len(series) < 2:
        continue

    # get the ratings of the movies
    franchise_df = df_10[series]
    ratings = []
    for title in series:
        ratings += [list(franchise_df[title].dropna())]

    if len(ratings) == 2:
        # use t test if only 2 movies
        stat, p_value = stats.ttest_ind(
            ratings[0],
            ratings[1], 
            equal_var=False, 
            nan_policy='omit'
        )

    else:
        # use ANOVA if more than 2 movies
        stat, p_value = stats.f_oneway(*ratings)

    is_significant = p_value < alpha
    if is_significant:
        significant_num += 1
        significant_movies.append(franchise)     

print("proportion of significant results:", significant_num / len(franchises))
print("significant movies are:", significant_movies)


# non-parametric tests
significant_num = 0
significant_movies = []

for franchise in franchises:
    # find the movies in the franchise
    series = [title for title in movie_titlles if franchise in title]

    # drop series with less than 2 movies
    if len(series) < 2:
        continue

    # get the ratings of the movies
    franchise_df = df_10[series]
    ratings = []
    for title in series:
        ratings += [list(franchise_df[title].dropna())]

    if len(ratings) == 2:
        # use u test if only 2 movies
        stat, p_value = stats.mannwhitneyu(
            ratings[0],
            ratings[1], 
            alternative='two-sided',
            nan_policy='omit'
        )

    else:
        # use kruskal H-test if more than 2 movies
        stat, p_value = stats.kruskal(*ratings)

    is_significant = p_value < alpha
    if is_significant:
        significant_num += 1  
        significant_movies.append(franchise)  

print("proportion of significant results:", significant_num / len(franchises))
print("significant movies are:", significant_movies)

########################################################################################################################

# Extra Credit

'''Extra Credit: We'll try figure out if there's any significant gender-based difference in 
ratings for a set of selected family-friendly movies. '''

keywords = ["Toy Story", "The Lion King", "Finding Nemo", "Shrek", "Aladdin", "The Jungle Book"]
family_friendly_titles = [title for title in df.columns if any(keyword in title for keyword in keywords)]

print(family_friendly_titles)
print(f"Number of family friendly movies: {len(family_friendly_titles)}" )

# Initialize Dict for gender-based ratings for selected animated movies
male_ratings_ff = {}
female_ratings_ff = {}

for title in family_friendly_titles:
    male_ratings_ff[title] = df[title][gender_data == 2].dropna()
    female_ratings_ff[title] = df[title][gender_data == 1].dropna()
    
#print(male_ratings_ff)
#print(female_ratings_ff)


male_means_ff = {title: ratings.mean() for title, ratings in male_ratings_ff.items()}
female_means_ff = {title: ratings.mean() for title, ratings in female_ratings_ff.items()}

# Create a DataFrame for plotting
family_friendly_df = pd.DataFrame({
    'Movie': list(male_means_ff.keys()),
    'Male Ratings': list(male_means_ff.values()),
    'Female Ratings': list(female_means_ff.values())
})

family_friendly_melted = family_friendly_df.melt(id_vars='Movie', var_name='Gender', value_name='Average Rating')

# Average Ratings by Gender for Family-Friendly Movies
plt.figure()
sns.barplot(data=family_friendly_melted, x='Movie', y='Average Rating', hue='Gender')
plt.xticks(rotation=45)
plt.xlabel('Movie')
plt.ylabel('Average Rating')
plt.title('Average Ratings by Gender for Selected Family-Friendly Movies')
plt.legend(title='Gender')
plt.show()


# Using Mannwhitney test
gender_data = df.iloc[:, 474]
alpha = 0.005
significant_family_friendly_count = 0

for title in family_friendly_titles:
    male_ratings = male_ratings_ff[title]
    female_ratings = female_ratings_ff[title]
    
    # Perform test if both male and female ratings are available
    if not male_ratings.empty and not female_ratings.empty:
        stat, p_value = stats.mannwhitneyu(male_ratings, female_ratings, alternative='two-sided')
        
        # Check if result is significant
        if p_value < alpha:
            significant_family_friendly_count += 1
            
print(f"Number of significant movies: {significant_family_friendly_count}")


#proportion
proportion_ff = significant_family_friendly_count / len(family_friendly_titles)
print(f"Proportion of selected family friendly movies with significant gender-based rating difference: {proportion_ff:.4f}")



