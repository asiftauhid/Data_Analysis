# NYU 2024 Fall DSGA 1001 Captone Project
# Author: Asif Tauhid, Haojie Cai, Xiaokan Tian

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import os

seed = 17151398
np.random.seed(seed)
alpha_level = 0.005


# MACROS
BASE_RATING_NUM = 1
TAG_NORMALIZE = 1   # normalization: 0: no standarlization, 1: standarlization, 2: standardlization + bias
RATING_SCORE = 0.00001


def check_var_similar(df_1, df_2): 
    levene_stat, levene_p = stats.levene(df_1, df_2, center='mean')
    if levene_p > alpha_level: 
        return True
    else: 
        return False
    
def cohens_d(df1, df2):
    pooled_std = np.sqrt((np.std(df1, ddof=1)**2 + np.std(df2, ddof=1)**2) / 2)
    d = (np.mean(df1) - np.mean(df2))/ pooled_std
    se = np.sqrt((len(df1) + len(df2)) / (len(df1) * len(df2)) + d**2 / (2 * (len(df1) + len(df2))))
    return d, se

def var_d(df1, df2):
    pooled_var = (np.std(df1, ddof=1)**2 + np.std(df2, ddof=1)**2) / 2
    d = abs((np.var(df1) - np.var(df2))) / pooled_var
    return d


## STEP 0. Preprocessing
df_num_columns = [
    "Average Rating", "Average Difficulty", "Number of Ratings",
    "Received a Pepper", "Proportion Retaking Class",
    "Ratings from Online Classes", "Male Gender", "Female Gender"
]

df_qual_columns = ["Major/Field", "University", "US State"]

df_tag_columns = [
    "Tough Grader", "Good Feedback", "Respected", "Lots to Read",
    "Participation Matters", "Don't Skip Class", "Lots of Homework",
    "Inspirational", "Pop Quizzes", "Accessible", "So Many Papers",
    "Clear Grading", "Hilarious", "Test Heavy", "Graded by Few Things",
    "Amazing Lectures", "Caring", "Extra Credit", "Group Projects",
    "Lecture Heavy"
]

current_path = os.path.dirname(os.path.abspath(__file__))
num_path = os.path.join(current_path, 'rmpCapstoneNum.csv')
qual_path = os.path.join(current_path, 'rmpCapstoneQual.csv')
tag_path = os.path.join(current_path, 'rmpCapstoneTags.csv')

# load data
df_num = pd.read_csv(num_path, names=df_num_columns, header = None)
df_qual = pd.read_csv(qual_path, names=df_qual_columns, header = None)
df_tag = pd.read_csv(tag_path, names=df_tag_columns, header=None)

# merge data
merged_data = pd.concat([df_num.reset_index(drop=True),
                         df_qual.reset_index(drop=True),
                         df_tag.reset_index(drop=True)], axis=1)

# check for missing values
count_na = merged_data.isna().sum()
# print(count_na)    # show missing values in each col

# check the number of ratings 
# merged_data["Number of Ratings"].describe()

# drop rows with no ratings (Number of Ratings = 0)
merged_data = merged_data[merged_data['Number of Ratings'] >= BASE_RATING_NUM]

# normalization
# for tags, we know that each tag might be competitve to each other (since only 3 tags allowed)
# Goal: minimize this competitive effect and normalize the tag columns
# RATING_SCORE: for professor with same ratio of tag/rating_number, the one receiving more tags should have higher score, so we add this to show this effect
if TAG_NORMALIZE == 0:  # without standardlization
    for tag in df_tag_columns: 
        merged_data[tag] = merged_data[tag] / merged_data["Number of Ratings"]
elif TAG_NORMALIZE == 1:    # with standardlization
    for tag in df_tag_columns:
        merged_data[tag] = merged_data[tag] / merged_data["Number of Ratings"]
        merged_data[tag] = (merged_data[tag] - merged_data[tag].mean()) / merged_data[tag].std()    # standardlization
elif TAG_NORMALIZE == 2:    # with standardlization and rating score
    for tag in df_tag_columns:
        merged_data[tag] = merged_data[tag] / merged_data["Number of Ratings"] + merged_data[tag] * RATING_SCORE
        merged_data[tag] = (merged_data[tag] - merged_data[tag].mean()) / merged_data[tag].std()    # standardlization


## QUESTION 1
# preprocessing
print("\n")
print("QUESTION 1: START")

male_prof = merged_data[(merged_data['Male Gender'] == 1) & (merged_data['Female Gender'] == 0) ]
female_prof = merged_data[(merged_data['Female Gender'] == 1) & (merged_data['Male Gender'] == 0) ]

male_rating = male_prof['Average Rating']
female_rating = female_prof['Average Rating']


# check whether sample size n is large enough
#print("The size of male professor sample: ", len(male_prof))
#rint("The size of female professor sample: ", len(female_prof))

# normal case
gender_similar = check_var_similar(male_rating, female_rating)

# Welch's t-test
t_stat, p_value = stats.ttest_ind(
    male_rating,
    female_rating,
    equal_var=gender_similar,
    nan_policy='omit'
)

is_significant = p_value < alpha_level
print("test statistic for professor:", t_stat)
print("p-value for professor:", p_value)
print("is significant for professor:", is_significant)


plot_data = pd.DataFrame({
    'Number_of_Ratings': np.concatenate([male_prof['Number of Ratings'], female_prof['Number of Ratings']]),
    'Gender': ['Male'] * len(male_prof) + ['Female'] * len(female_prof)
})

sns.violinplot(x='Gender', y='Number_of_Ratings', data=plot_data)
plt.title('Distribution of Number of Ratings by Gender')
plt.show()

print("-------------------------------------------")
print("male professor info:", male_prof['Number of Ratings'].describe())
print("-------------------------------------------")
print("female professor info:", female_prof['Number of Ratings'].describe())
print("-------------------------------------------")

# ks test to check whether the distribution of number of ratings is similar
ks_stat, ks_p_value = stats.ks_2samp(male_prof['Number of Ratings'],female_prof['Number of Ratings'])
print("ks statistic for professor:", ks_stat)
print("p-value for professor:", ks_p_value)
print("is significant for professor:", ks_p_value < alpha_level)

# Since there's significant difference, we have to group the data by their teaching experience

# new professor
male_new = male_prof[male_prof['Number of Ratings'] <= 3]
female_new = female_prof[female_prof['Number of Ratings'] <= 3]

# experienced professor
male_experienced = male_prof[male_prof['Number of Ratings'] <= 6]
female_experienced = female_prof[female_prof['Number of Ratings'] <= 6]

# senior professor (for senior group, since their teaching experience is large enough, we don't need to filter)
male_senior = male_prof[male_prof['Number of Ratings'] > 6]
female_senior = female_prof[female_prof['Number of Ratings'] > 6]

experience_data = {
    "male_new": male_new,
    "female_new": female_new,
    "male_experienced": male_experienced,
    "female_experienced": female_experienced,
    "male_senior": male_senior,
    "female_senior": female_senior
}
experience_label = ['new', 'experienced', 'senior']

# check whether whether their variance is similar
for lable in experience_label:
    print("variance of", lable, "professor similar:", check_var_similar(experience_data[f"male_{lable}"]['Average Rating'], experience_data[f"female_{lable}"]['Average Rating']))

plot_data = pd.DataFrame({
    'Average Rating': np.concatenate([
        experience_data['male_new']['Average Rating'], 
        experience_data['female_new']['Average Rating'],
        experience_data['male_experienced']['Average Rating'],
        experience_data['female_experienced']['Average Rating'],
        experience_data['male_senior']['Average Rating'],
        experience_data['female_senior']['Average Rating']
    ]),
    'Gender': (
        ['Male'] * len(experience_data['male_new']) +
        ['Female'] * len(experience_data['female_new']) +
        ['Male'] * len(experience_data['male_experienced']) +
        ['Female'] * len(experience_data['female_experienced']) +
        ['Male'] * len(experience_data['male_senior']) +
        ['Female'] * len(experience_data['female_senior'])
    ),
    'Experience Level': (
        ['New'] * (len(experience_data['male_new']) + len(experience_data['female_new'])) +
        ['Experienced'] * (len(experience_data['male_experienced']) + len(experience_data['female_experienced'])) +
        ['Senior'] * (len(experience_data['male_senior']) + len(experience_data['female_senior']))
    )
})

# Draw boxplot
plt.figure(figsize=(10, 6))
boxplot = sns.boxplot(
    x='Experience Level', 
    y='Average Rating', 
    hue='Gender', 
    data=plot_data
)

plt.title('Average Rating by Gender and Experience Level')
plt.ylabel('Average Rating')
plt.xlabel('Experience Level')
plt.legend(title='Gender')
plt.show()


# for each experience group, check whether the average rating is significantly different

for label in experience_label:
    # Welch t-test
    t_stat, p_value = stats.ttest_ind(
        experience_data[f"male_{label}"]['Average Rating'],
        experience_data[f"female_{label}"]['Average Rating'],
        equal_var=False,
        nan_policy='omit'
    )

    is_significant = p_value < alpha_level
    print(f"test statistic for {label} professor:", t_stat)
    print(f"p-value for {label} professor:", p_value)
    print(f"is significant for {label} professor:", is_significant)
    if is_significant:
        if t_stat > 0:
            print(f"For {label} professor, male professor has higher average rating")
        else:
            print(f"For {label} professor, female professor has higher average rating")
    print("-------------------------------------------")


print("\n\n\n")

## QUESTION 2
print("QUESTION 2: START")

# visualize the distribution of number of ratings in violin plot
plot_data = pd.DataFrame({
    'Average Rating': np.concatenate([male_rating, female_rating]),
    'Gender': ['Male'] * len(male_rating) + ['Female'] * len(female_rating)
})

sns.boxplot(data= plot_data, x = 'Gender', y = 'Average Rating', notch=False, width=0.5)
plt.title("Boxplot of Average Ratings by Gender")
plt.show()


# Levene's test for variance
levene_stat, levene_p = stats.levene(male_rating, female_rating, center='mean')
is_significant = levene_p < alpha_level
print("Levene's test statistic:", levene_stat)
print("p-value:", levene_p)
print("is significant:", is_significant)


# check variance
male_var = np.var(male_rating)
female_var = np.var(female_rating)
print("Variance of Male", male_var)
print("Variance of female", female_var)

for label in experience_label:
    # Welch t-test
    levene_stat, levene_p = stats.levene(
        experience_data[f"male_{label}"]['Average Rating'],
        experience_data[f"female_{label}"]['Average Rating'],
        center='mean'
    )

    is_significant = levene_p < alpha_level
    print(f"test statistic for {label} professor:", levene_stat)
    print(f"p-value for {label} professor:", levene_p)
    print(f"is significant for {label} professor:", is_significant)
    print("-------------------------------------------")



print("\n\n\n")

## QUESTION 3
print("QUESTION 3: START")

# effect size of gender bias in average rating by cohen's d
d, se = cohens_d(male_rating, female_rating)
upper_bound = d+ 1.96 * se
lower_bound = d- 1.96 * se
print("Cohends d is:",d)
print("confidence interval is:",[lower_bound, upper_bound]) 

np.random.seed(seed=seed)  # For reproducibility
boot_d = []
for _ in range(10000):
    male_sample = np.random.choice(male_rating, size=len(male_rating), replace=True)
    female_sample = np.random.choice(female_rating, size=len(female_rating), replace=True)

    boot_d.append(cohens_d(male_sample, female_sample)[0])
ci_cohen_d = np.percentile(boot_d, [2.5, 97.5])

print(f"95% CI for Cohen's d: {ci_cohen_d}")

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(boot_d, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)

# Add vertical lines for the 95% CI
plt.axvline(ci_cohen_d[0], color='red', linestyle='dashed', linewidth=2, label=f'Lower 95% CI ({ci_cohen_d[0]:.2f})')
plt.axvline(ci_cohen_d[1], color='red', linestyle='dashed', linewidth=2, label=f'Upper 95% CI ({ci_cohen_d[1]:.2f})')

# Add labels, title, and legend
plt.title('Bootstrapped Distribution of Cohen\'s d with 95% CI', fontsize=14)
plt.xlabel('Cohen\'s d', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

# Show the plot
plt.show()

d = var_d(male_rating, female_rating)
print("variance_d is:",d)

np.random.seed(seed=seed)  # For reproducibility
boot_d = []
for _ in range(10000):
    male_sample = np.random.choice(male_rating, size=len(male_rating), replace=True)
    female_sample = np.random.choice(female_rating, size=len(female_rating), replace=True)

    boot_d.append(var_d(male_sample, female_sample))
ci_cohen_d = np.percentile(boot_d, [2.5, 97.5])

print(f"95% CI for Cohen's d: {ci_cohen_d}")

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(boot_d, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)


# Add labels, title, and legend
plt.title('Bootstrapped Distribution of Var\'s d with 95% CI', fontsize=14)
plt.xlabel('Var\'s d', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

# Show the plot
plt.show()

f_ratio = np.var(male_rating) / np.var(female_rating)
print("f_ratio is:",f_ratio)


np.random.seed(seed=seed)  # For reproducibility
boot_d = []
for _ in range(10000):
    male_sample = np.random.choice(male_rating, size=len(male_rating), replace=True)
    female_sample = np.random.choice(female_rating, size=len(female_rating), replace=True)

    boot_d.append(np.var(male_sample) / np.var(female_sample))
ci_cohen_d = np.percentile(boot_d, [2.5, 97.5])

print(f"95% CI for f-ratio: {ci_cohen_d}")

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(boot_d, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)


# Add labels, title, and legend
plt.title('Bootstrapped Distribution of f-ratio with 95% CI', fontsize=14)
plt.xlabel('f-ratio', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

# Show the plot
plt.show()


print("\n\n\n")


## QUESTION 4
print("QUESTION 4: START")
result = []

# Iterates through each tag
for tag in df_tag_columns:
    male_tag = male_prof[tag]
    female_tag = female_prof[tag]

    variance_similar = check_var_similar(male_tag, female_tag)

    # Welch's t-test
    t_stat, p_value = stats.ttest_ind(
        male_tag,
        female_tag,
        equal_var=variance_similar,
        nan_policy='omit'
    )

    is_significant = p_value < alpha_level

    result.append({ # store the results
        "tag": tag,
        "t_stat": t_stat,
        "p_value": p_value,
        "is_significant": is_significant
    })

    if (is_significant):
        print(f"tag {tag} is significant")

result_df = pd.DataFrame(result)
result_df = result_df.sort_values(by='p_value')

top = result_df.head(3)
bottom = result_df.tail(3)

print("-------------------------------------------")
print("Top 3 significant tags:")
print(top)
print("-------------------------------------------")
print("Bottom 3 significant tags:")
print(bottom)

# Bar Chart: All tags sorted by p-value
plt.figure(figsize=(10, 6))
sorted_tags = result_df.sort_values(by='p_value')
plt.bar(sorted_tags['tag'], sorted_tags['p_value'], color=['red' if sig else 'blue' for sig in sorted_tags['is_significant']])
plt.axhline(y=alpha_level, color='green', linestyle='--', label='Significance Level')
plt.xticks(rotation=90)
plt.title("P-values for Tags (Sorted)")
plt.xlabel("Tags")
plt.ylabel("P-value")
plt.legend()
plt.tight_layout()
plt.show()


print("\n\n\n")

##  QUESTION 5
print("QUESTION 5: START")

male_difficulty = male_prof['Average Difficulty']
female_difficulty = female_prof['Average Difficulty']

gender_similar = check_var_similar(male_difficulty, female_difficulty)

# Welch's t-test
t_stat, p_value = stats.ttest_ind(
    male_difficulty,
    female_difficulty,
    equal_var=gender_similar,
    nan_policy='omit'
)

is_significant = p_value < alpha_level
print("test statistic for professor:", t_stat)
print("p-value for professor:", p_value)
print("is significant for professor:", is_significant)


# Mann-Whitney U Test
u_stat, p_value = stats.mannwhitneyu(male_difficulty, female_difficulty, alternative='two-sided')

# Test statistics and p-value
print(f"U-Statistic = {u_stat}")
print(f"p-value = {p_value}")

# Checking significance
if p_value < alpha_level:
    print("The difference in average difficulty is statistically significant.")
else:
    print("The difference in average difficulty is not statistically significant.")


print("\n\n\n")

## QUESTION 6
print("QUESTION 6: START")
np.random.seed(seed=seed)  # For reproducibility
boot_d = []
for _ in range(10000):
    male_sample = np.random.choice(male_difficulty, size=len(male_difficulty), replace=True)
    female_sample = np.random.choice(female_difficulty, size=len(female_difficulty), replace=True)

    boot_d.append(cohens_d(male_sample, female_sample)[0])
ci_cohen_d = np.percentile(boot_d, [2.5, 97.5])

print(f"95% CI for Cohen's d: {ci_cohen_d}")

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(boot_d, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)

# Add vertical lines for the 95% CI
plt.axvline(ci_cohen_d[0], color='red', linestyle='dashed', linewidth=2, label=f'Lower 95% CI ({ci_cohen_d[0]:.2f})')
plt.axvline(ci_cohen_d[1], color='red', linestyle='dashed', linewidth=2, label=f'Upper 95% CI ({ci_cohen_d[1]:.2f})')

# Add labels, title, and legend
plt.title('Bootstrapped Distribution of Cohen\'s d with 95% CI', fontsize=14)
plt.xlabel('Cohen\'s d', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

# Show the plot
plt.show()

print("\n\n\n")

## QUESTION 7
print("QUESTION 7: START")

filtered_data_40k = merged_data[merged_data["Number of Ratings"] >= 5]
filtered_data_40k = filtered_data_40k[filtered_data_40k['Proportion Retaking Class'] >= 0]

X = filtered_data_40k[["Average Difficulty", "Number of Ratings", "Proportion Retaking Class",
          "Ratings from Online Classes", "Male Gender", "Female Gender"]]
y = filtered_data_40k["Average Rating"]

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# linear regression
lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_val)

# calculate r2, MSE
r2_lm = r2_score(y_val, y_pred)
rmse_lm = mean_squared_error(y_val, y_pred, squared=False)
print("lm R2: ", r2_lm)
print("lm RMSE: ", rmse_lm)

# ridge regression
alphas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0]
ridge_val_mse = []
ridge_r2 = []

for alpha in alphas:
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)
    
    y_val_pred_ridge = ridge_model.predict(X_val)
    
    ridge_val_mse.append(mean_squared_error(y_val, y_val_pred_ridge))
    ridge_r2.append(r2_score(y_val, y_val_pred_ridge))
    
    
best_alpha_ridge = 0.01
print(ridge_val_mse, ridge_r2)
plt.figure(figsize=(10, 6))
plt.plot(alphas, ridge_val_mse, label='Validation MSE', marker='o')
plt.plot(alphas, ridge_r2, label='Validation R2', marker='o')
plt.xscale('log')  # Log scale for alpha
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Mean Squared Error')
plt.title('Effect of Ridge Regularization on MSE')
plt.legend()
plt.grid(True)
plt.show()

ridge_model = Ridge(alpha=best_alpha_ridge)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_val)
betas_ridge = ridge_model.coef_

plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred_ridge, alpha=0.7, color='blue')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', linestyle='--')
plt.title("Predicted vs Actual Values (Ridge Regression)")
plt.xlabel("Actual Values (y_val)")
plt.ylabel("Predicted Values (y_pred_ridge)")
plt.grid(True)
plt.tight_layout()
plt.show()

feature_names = X.columns
plt.bar(feature_names, betas_ridge)
plt.title(f'Coefficients Ridge (alpha={best_alpha_ridge})')
plt.xlabel('Feature Index')
plt.xticks(rotation=45, ha='center')
plt.ylabel('Coefficient Value')
plt.show()

# LASSO regression
lasso_val_mse = []
lasso_r2 = []

alphas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0]

for alpha in alphas:
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train, y_train)

    y_val_pred_lasso = lasso_model.predict(X_val)

    lasso_val_mse.append(mean_squared_error(y_val, y_val_pred_lasso))
    lasso_r2.append(r2_score(y_val, y_val_pred_lasso))
    

best_alpha_lasso = 0.01
lasso_val_mse, lasso_r2

plt.figure(figsize=(10, 6))
plt.plot(alphas, lasso_val_mse, label='Validation MSE', marker='o')
plt.plot(alphas, lasso_r2, label='Validation R2', marker='o')
plt.xscale('log')  # Log scale for alpha
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Mean Squared Error')
plt.title('Effect of Lasso Regularization on MSE')
plt.legend()
plt.grid(True)
plt.show()


print("\n\n\n")


## QUESTION 8
print("QUESTION 8: START")

filtered_data_40k = merged_data[merged_data["Number of Ratings"] >= 5]

X = filtered_data_40k[[
    "Tough Grader", "Good Feedback", "Respected", "Lots to Read",
    "Participation Matters", "Don't Skip Class", "Lots of Homework",
    "Inspirational", "Pop Quizzes", "Accessible", "So Many Papers",
    "Clear Grading", "Hilarious", "Test Heavy", "Graded by Few Things",
    "Amazing Lectures", "Caring", "Extra Credit", "Group Projects",
    "Lecture Heavy"
]]
y = filtered_data_40k["Average Rating"]

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# ridge regression
alphas = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0]
ridge_val_mse = []
ridge_r2 = []

for alpha in alphas:
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)
    
    y_val_pred_ridge = ridge_model.predict(X_val)
    
    ridge_val_mse.append(mean_squared_error(y_val, y_val_pred_ridge))
    ridge_r2.append(r2_score(y_val, y_val_pred_ridge))

plt.figure(figsize=(10, 6))
plt.plot(alphas, ridge_val_mse, label='Validation MSE', marker='o')
plt.plot(alphas, ridge_r2, label='Validation R2', marker='o')
plt.xscale('log')  # Log scale for alpha
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Mean Squared Error')
plt.title('Effect of Ridge Regularization on MSE')
plt.legend()
plt.grid(True)
plt.show()

best_alpha_ridge = 0.001
ridge_model = Ridge(alpha=best_alpha_ridge)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_val)
betas_ridge = ridge_model.coef_

plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred_ridge, alpha=0.7, color='blue')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', linestyle='--')
plt.title("Predicted vs Actual Values (Ridge Regression)")
plt.xlabel("Actual Values (y_val)")
plt.ylabel("Predicted Values (y_pred_ridge)")
plt.grid(True)
plt.tight_layout()
plt.show()

feature_names = X.columns
plt.bar(feature_names, betas_ridge)
plt.title(f'Coefficients Ridge (alpha={best_alpha_ridge})')
plt.xlabel('Feature Index')
plt.xticks(rotation=70, ha='center')
plt.ylabel('Coefficient Value')


lasso_val_mse = []
lasso_r2 = []

alphas = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0]

for alpha in alphas:
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train, y_train)

    y_val_pred_lasso = lasso_model.predict(X_val)

    lasso_val_mse.append(mean_squared_error(y_val, y_val_pred_lasso))
    lasso_r2.append(r2_score(y_val, y_val_pred_lasso))
    
lasso_val_mse, lasso_r2

plt.figure(figsize=(10, 6))
plt.plot(alphas, lasso_val_mse, label='Validation MSE', marker='o')
plt.plot(alphas, lasso_r2, label='Validation R2', marker='o')
plt.xscale('log')  # Log scale for alpha
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Mean Squared Error')
plt.title('Effect of Lasso Regularization on MSE')
plt.legend()
plt.grid(True)
plt.show()


best_alpha_lasso = 0.01
lasso_model = Lasso(alpha=best_alpha_lasso)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_val)
betas_lasso = lasso_model.coef_

plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred_lasso, alpha=0.7, color='blue')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', linestyle='--')
plt.title("Predicted vs Actual Values (Lasso Regression)")
plt.xlabel("Actual Values (y_val)")
plt.ylabel("Predicted Values (y_pred_lasso)")
plt.grid(True)
plt.tight_layout()
plt.show()


plt.bar(range(len(betas_lasso)), betas_lasso)
plt.title(f'Coefficients Lasso (alpha={best_alpha_lasso})')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')


print("\n\n\n")

## QUESTION 9
print("QUESTION 9: START")

# Q9
X = filtered_data_40k[[
    "Tough Grader", "Good Feedback", "Respected", "Lots to Read",
    "Participation Matters", "Don't Skip Class", "Lots of Homework",
    "Inspirational", "Pop Quizzes", "Accessible", "So Many Papers",
    "Clear Grading", "Hilarious", "Test Heavy", "Graded by Few Things",
    "Amazing Lectures", "Caring", "Extra Credit", "Group Projects",
    "Lecture Heavy"
]]
y = filtered_data_40k["Average Difficulty"]

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# ridge regression
alphas = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0]
ridge_val_mse = []
ridge_r2 = []

for alpha in alphas:
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)
    
    y_val_pred_ridge = ridge_model.predict(X_val)
    
    ridge_val_mse.append(mean_squared_error(y_val, y_val_pred_ridge))
    ridge_r2.append(r2_score(y_val, y_val_pred_ridge))

plt.figure(figsize=(10, 6))
plt.plot(alphas, ridge_val_mse, label='Validation MSE', marker='o')
plt.plot(alphas, ridge_r2, label='Validation R2', marker='o')
plt.xscale('log')  # Log scale for alpha
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Mean Squared Error')
plt.title('Effect of Ridge Regularization on MSE')
plt.legend()
plt.grid(True)
plt.show()

best_alpha_ridge = 0.001
ridge_model = Ridge(alpha=best_alpha_ridge)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_val)
betas_ridge = ridge_model.coef_

plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred_ridge, alpha=0.7, color='blue')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', linestyle='--')
plt.title("Predicted vs Actual Values (Ridge Regression)")
plt.xlabel("Actual Values (y_val)")
plt.ylabel("Predicted Values (y_pred_ridge)")
plt.grid(True)
plt.tight_layout()
plt.show()

feature_names = X.columns
plt.bar(feature_names, betas_ridge)
plt.title(f'Coefficients Ridge (alpha={best_alpha_ridge})')
plt.xlabel('Feature Index')
plt.xticks(rotation=70, ha='center')
plt.ylabel('Coefficient Value')


lasso_val_mse = []
lasso_r2 = []

alphas = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0]

for alpha in alphas:
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train, y_train)

    y_val_pred_lasso = lasso_model.predict(X_val)

    lasso_val_mse.append(mean_squared_error(y_val, y_val_pred_lasso))
    lasso_r2.append(r2_score(y_val, y_val_pred_lasso))


plt.figure(figsize=(10, 6))
plt.plot(alphas, lasso_val_mse, label='Validation MSE', marker='o')
plt.plot(alphas, lasso_r2, label='Validation R2', marker='o')
plt.xscale('log')  # Log scale for alpha
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Mean Squared Error')
plt.title('Effect of Lasso Regularization on MSE')
plt.legend()
plt.grid(True)
plt.show()


best_alpha_lasso = 0.01
lasso_model = Lasso(alpha=best_alpha_lasso)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_val)
betas_lasso = lasso_model.coef_

plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred_lasso, alpha=0.7, color='blue')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', linestyle='--')
plt.title("Predicted vs Actual Values (Lasso Regression)")
plt.xlabel("Actual Values (y_val)")
plt.ylabel("Predicted Values (y_pred_lasso)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.bar(range(len(betas_lasso)), betas_lasso)
plt.title(f'Coefficients Lasso (alpha={best_alpha_lasso})')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')

print("\n\n\n")

## QUESTION 10
print("QUESTION 10: START")

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

filtered_data_40k = merged_data[merged_data["Number of Ratings"] >= 5]
filtered_data_40k = filtered_data_40k[filtered_data_40k['Proportion Retaking Class'] >= 0]

# exclude Received a Pepper and merge all the other columns
feature_columns = [col for col in df_num_columns if col != "Received a Pepper"] + df_tag_columns

filtered_data_40k[feature_columns]


X = filtered_data_40k[feature_columns]
y = filtered_data_40k['Received a Pepper']
print(feature_columns)

# Use SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# train a random forest classifier
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# AUROC and class predictions
y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Compute AUROC
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"AUROC: {roc_auc}")

# Classification Report
print(classification_report(y_test, y_pred))

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUROC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Random Guess')
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_pred, cmap='coolwarm', alpha=0.7)
plt.colorbar(scatter, label='Predicted Class')
plt.xlabel(feature_columns[0])
plt.ylabel(feature_columns[1])
plt.title("Scatter Plot of Predicted Classes")
plt.show()

print("\n\n\n")

## EXTRA CREDIT
# Proportion of Peppers in terms of states
#Calculate global statistics
global_mean = merged_data['Received a Pepper'].mean()
state_grouped = merged_data.groupby('US State')['Received a Pepper']
global_variance = state_grouped.mean().var()

# Estimate alpha and beta dynamically
alpha = global_mean * (global_mean * (1 - global_mean) / global_variance - 1)
beta = (1 - global_mean) * (global_mean * (1 - global_mean) / global_variance - 1)

# Group by state and calculate relevant stats
state_data = (
    merged_data.groupby('US State')['Received a Pepper']
    .agg(['sum', 'count'])  # Sum is the number of "peppers", count is the total professors
    .reset_index()
    .rename(columns={'sum': 'Total Peppers', 'count': 'Total Professors'})
)

# Filter states with at least 10 professors
state_data = state_data[state_data['Total Professors'] >= 10]

# Calculate Bayesian smoothed proportion
state_data['Proportion With Pepper'] = (
    (state_data['Total Peppers'] + alpha * global_mean) /
    (state_data['Total Professors'] + alpha + beta)
)

# Sort for visualization
state_data = state_data.sort_values(by='Proportion With Pepper', ascending=False)
print(f"National Average = {state_data['Proportion With Pepper'].mean()}")
print(state_data)

# Bar plot for Bayesian smoothed proportions
plt.figure(figsize=(12, 8))
plt.bar(state_data['US State'], state_data['Proportion With Pepper'], color='skyblue')
plt.title('Bayesian Smoothed Proportion of Professors Receiving Peppers by State (Filtered)', fontsize=16)
plt.xlabel('US State', fontsize=14)
plt.ylabel('Proportion With Pepper (Bayesian Smoothed)', fontsize=14)
plt.xticks(rotation=90, fontsize=10)
plt.tight_layout()
plt.show()

# Calculate correlations between tags and pepper
correlations = merged_data[df_tag_columns + ['Received a Pepper']].corr()['Received a Pepper'].drop('Received a Pepper')

# Sort correlations
sorted_correlations = correlations.sort_values(ascending=False)
print(sorted_correlations)

# Visualize the correlations in a bar chart
plt.figure(figsize=(12, 8))
sorted_correlations.plot(kind='bar', color='skyblue')
plt.title('Correlation of Tags with Receiving a Pepper', fontsize=16)
plt.xlabel('Tags', fontsize=14)
plt.ylabel('Correlation Coefficient', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.show()








