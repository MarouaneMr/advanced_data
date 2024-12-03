# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder

# Step 2: Load Data
skills = pd.read_csv('skills.csv')
job_skills = pd.read_csv('job_skills.csv')
industries = pd.read_csv('industries.csv')
job_industries = pd.read_csv('job_industries.csv')
postings = pd.read_csv('postings.csv')
salaries_cleaned = pd.read_csv('salaries_cleaned.csv')
employee_counts = pd.read_csv('new_employee_counts.csv')
industries_with_clusters = pd.read_csv('industries_with_clusters.csv')
companies_cleaned = pd.read_csv('companies_cleaned.csv')

# Step 3: Merge and Prepare Data
# Merge job_skills with skills to get skill_name
job_skills_merged = job_skills.merge(skills, on='skill_abr', how='inner')

# Merge job_skills_merged with job_industries
job_data = job_skills_merged.merge(job_industries, on='job_id', how='inner')

# Merge with industries
job_data = job_data.merge(industries, on='industry_id', how='inner')

# Merge with postings to get title, description, and salaries
job_data = job_data.merge(postings[['job_id', 'company_id', 'title', 'description', 
                                    'max_salary', 'med_salary', 'min_salary']], on='job_id', how='left')

# Merge additional datasets
job_data = job_data.merge(salaries_cleaned, on='job_id', how='left')
job_data = job_data.merge(employee_counts, on='company_id', how='left')
job_data = job_data.merge(industries_with_clusters, on='industry_id', how='left')
job_data = job_data.merge(companies_cleaned, on='company_id', how='left')

# Rename and clean columns for consistency
if 'description_x' in job_data.columns:
    job_data.rename(columns={'description_x': 'description'}, inplace=True)
if 'description_y' in job_data.columns:
    job_data.drop(columns=['description_y'], inplace=True)

# Drop rows with missing critical values
job_data.dropna(subset=['title', 'description', 'skill_name'], inplace=True)

# Step 4: Classify Roles
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

# Step 5: Skill Clustering
# Encode skills numerically
skill_encoded = job_data['skill_name'].astype('category').cat.codes
job_data['skill_encoded'] = skill_encoded

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
job_data['skill_cluster'] = kmeans.fit_predict(job_data[['skill_encoded']])

# Define descriptive labels for clusters
cluster_labels = {
    0: 'Rare Skills',
    1: 'Specialized Skills',
    2: 'General Skills',
    3: 'High-Demand Skills',
    4: 'Advanced Skills'
}
job_data['skill_cluster_label'] = job_data['skill_cluster'].map(cluster_labels)

# Step 6: Visualize Skill Clusters by Role
skills_by_role = job_data.groupby(['role_type', 'skill_cluster_label']).size().unstack(fill_value=0)
skills_by_role_percentage = skills_by_role.div(skills_by_role.sum(axis=1), axis=0) * 100

skills_by_role_percentage.plot(kind='bar', stacked=True, figsize=(10, 6), title='Skill Clusters by Role Type')
plt.xlabel('Role Type')
plt.ylabel('Percentage')
plt.legend(title='Skill Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Step 7: Visualize Skill Clusters by Top Industries
top_5_industries = job_data['industry_name_x'].value_counts().head(5).index
filtered_data = job_data[job_data['industry_name_x'].isin(top_5_industries)]

skills_by_industry = filtered_data.groupby(['industry_name_x', 'skill_cluster_label']).size().unstack(fill_value=0)
skills_by_industry_percentage = skills_by_industry.div(skills_by_industry.sum(axis=1), axis=0) * 100

skills_by_industry_percentage.plot(kind='bar', stacked=True, figsize=(12, 7), title='Skill Clusters by Top Industries')
plt.xlabel('Industry')
plt.ylabel('Percentage')
plt.legend(title='Skill Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Step 8: Regression Analysis
# Remove rows with missing salary values
job_data = job_data.dropna(subset=['med_salary'])

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

# Step 9: Save Results
job_data.to_csv('job_data_with_clusters_and_analysis.csv', index=False)
