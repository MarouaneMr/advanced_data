# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load Data
skills = pd.read_csv('/mnt/data/skills.csv')
job_skills = pd.read_csv('/mnt/data/job_skills.csv')
industries = pd.read_csv('/mnt/data/industries.csv')
job_industries = pd.read_csv('/mnt/data/job_industries.csv')
postings = pd.read_csv('/path/to/postings.csv')  # Correctly load postings as CSV

# Step 3: Data Merging
# Merge job_skills with skills
job_skills_merged = job_skills.merge(skills, on='skill_abr', how='inner')

# Merge job_skills_merged with job_industries
job_data = job_skills_merged.merge(job_industries, on='job_id', how='inner')

# Merge with industries
job_data = job_data.merge(industries, on='industry_id', how='inner')

# Merge with job postings for job title, description, and salary info
job_data = job_data.merge(postings[['job_id', 'title', 'description', 'max_salary', 'med_salary', 'min_salary']], on='job_id', how='inner')

# Step 4: Handle Missing Data
# Fill missing salaries with median values
job_data['max_salary'] = job_data['max_salary'].fillna(job_data['max_salary'].median())
job_data['med_salary'] = job_data['med_salary'].fillna(job_data['med_salary'].median())
job_data['min_salary'] = job_data['min_salary'].fillna(job_data['min_salary'].median())

# Drop rows with missing critical values
job_data.dropna(subset=['title', 'description', 'skill_name'], inplace=True)

# Step 5: Classify Jobs as Technical or Non-Technical
def classify_role(title, description):
    tech_keywords = ['engineer', 'developer', 'scientist', 'programmer', 'technician', 'analyst']
    non_tech_keywords = ['manager', 'sales', 'hr', 'assistant', 'support']

    title = str(title).lower()
    description = str(description).lower()

    if any(keyword in title for keyword in tech_keywords) or any(keyword in description for keyword in tech_keywords):
        return 'Technical'
    elif any(keyword in title for keyword in non_tech_keywords) or any(keyword in description for keyword in non_tech_keywords):
        return 'Non-Technical'
    else:
        return 'Other'

job_data['role_type'] = job_data.apply(lambda row: classify_role(row['title'], row['description']), axis=1)

# Step 6: TF-IDF Vectorization for Skill Names
vectorizer = TfidfVectorizer(stop_words='english')
skill_vectors = vectorizer.fit_transform(job_data['skill_name'])

# Apply K-Means Clustering
num_clusters = 5  # Adjust based on exploration
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
job_data['skill_cluster'] = kmeans.fit_predict(skill_vectors)

# Step 7: Analyze Skill Clusters Across Industries and Roles
# Overall Distribution
overall_cluster_distribution = job_data['skill_cluster'].value_counts(normalize=True) * 100

# Skill Clusters by Industry
skills_by_industry_percentage = job_data.groupby(['industry_name', 'skill_cluster']).size().unstack(fill_value=0)
skills_by_industry_percentage = skills_by_industry_percentage.div(skills_by_industry_percentage.sum(axis=1), axis=0) * 100

# Skill Clusters by Role Type
skills_by_role_percentage = job_data.groupby(['role_type', 'skill_cluster']).size().unstack(fill_value=0)
skills_by_role_percentage = skills_by_role_percentage.div(skills_by_role_percentage.sum(axis=1), axis=0) * 100

# Visualize Results
skills_by_role_percentage.T.plot(kind='bar', figsize=(12, 6), title='Skill Clusters: Technical vs. Non-Technical Roles')
plt.xlabel('Skill Clusters')
plt.ylabel('Percentage')
plt.show()

skills_by_industry_percentage.T.plot(kind='bar', stacked=True, figsize=(15, 8), title='Skill Clusters Across Industries')
plt.xlabel('Skill Clusters')
plt.ylabel('Percentage')
plt.legend(title='Industries', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Step 8: Regression Analysis (Salary vs. Skill Clusters)
# Encode role type for regression
encoder = LabelEncoder()
job_data['role_type_encoded'] = encoder.fit_transform(job_data['role_type'])

# Prepare features and target
X = job_data[['skill_cluster', 'role_type_encoded']]
y = job_data['med_salary']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Linear Regression Model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Evaluate with Cross-Validation
cv_scores = cross_val_score(regressor, X, y, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)

print("Cross-Validation RMSE:", cv_rmse.mean())

# Save Results
job_data.to_csv('job_data_with_clusters_and_analysis.csv', index=False)

Added job skills clustering and role analysis script
