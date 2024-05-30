import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import warnings
import umap

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN

#Import data
df_health = pd.read_csv('/Users/asifislam/Desktop/Northeastern Studies/Spring 2024/DS5230/Actual Project/Healthcare Providers.csv')


# Selecting specific features for summary statistics
selected_features = ['Number of Services', 'Number of Distinct Medicare Beneficiary/Per Day Services', 'Number of Medicare Beneficiaries',
                     'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 'Average Medicare Payment Amount', 'Average Medicare Standardized Amount']

# Calculate summary statistics for the selected features
summary_stats_selected = df_health[selected_features].describe()



def remove_comma(x):
    return x.replace(",","")

for col in selected_features:
    df_health[col] = pd.to_numeric(df_health[col].apply(lambda x: remove_comma(x)),errors= "ignore")

print(df_health[selected_features].describe())



# Set the style of seaborn plots
sns.set_theme(style="whitegrid")



# Plot box plots for numerical variables
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_health[selected_features])
plt.title('Box Plot of Numerical Variables')
plt.xlabel('Variables')
plt.ylabel('Values')
plt.xticks(rotation=45, fontsize=5)
plt.show()


# Histogram
plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
for feature in selected_features:
    plt.hist(df_health[feature], bins=2, alpha=0.5, label=feature)  # Adjust the number of bins as needed
plt.title('Histograms of Selected Features')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# Find out missing values 
print(df_health.isnull().sum())


DropCols = ['index', 'National Provider Identifier',
       'Last Name/Organization Name of the Provider',
       'First Name of the Provider', 'Middle Initial of the Provider','Street Address 1 of the Provider',
       'Street Address 2 of the Provider','Zip Code of the Provider',"HCPCS Code"]

df = df_health.drop(DropCols, axis = 1)

#print(df.isnull().sum())

def Preprocessing(data):
    
    
    #1.Imputing Missing Values

    data["Credentials of the Provider"] = data["Credentials of the Provider"].fillna(data["Credentials of the Provider"].mode()[0])
    data["Gender of the Provider"] = data["Gender of the Provider"].fillna(data["Gender of the Provider"].mode()[0])
    

   #2.Binary Encoding.

    
    BEcols = [var for var in data.columns if data[var].dtype == "O"]
    
    for col in BEcols:
        encoder = ce.BinaryEncoder(cols = [col])
        dfbin = encoder.fit_transform(data[col])
        data = pd.concat([data,dfbin], axis = 1)
        del data[col]

    #3. One-Hot-Encoding

#     data = pd.get_dummies(data,drop_first = True)
    
 
    #4. Standardization
 
    data_columns = data.columns
    std = StandardScaler()
    data = std.fit_transform(data)
    data = pd.DataFrame(data, columns = data_columns)
    
    return data


df = Preprocessing(df)

#print(df.head)

# Perform PCA and calculate explained variance ratio
pca = PCA()
X_pca = pca.fit_transform(df)
explained_variance_ratio = pca.explained_variance_ratio_

# Calculate cumulative explained variance
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)



# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(explained_variance_ratio, marker='o')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Principal Components')
plt.xticks(np.arange(len(explained_variance_ratio)), np.arange(1, len(explained_variance_ratio) + 1), rotation=90)
plt.grid(True)
plt.show()

# Plot cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(cumulative_variance_ratio, marker='o')
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio by Principal Components')
plt.xticks(np.arange(len(cumulative_variance_ratio)), np.arange(1, len(cumulative_variance_ratio) + 1), rotation=90)
plt.grid(True)
plt.show()



final_pca = PCA(n_components=5)
df_pca = final_pca.fit_transform(df)


# Calculate the silhouette scores for different numbers of clusters
silhouette_scores = []
for n in range(2, 11):  # Silhouette score is not defined for n=1
    kmeans = KMeans(n_clusters=n, random_state=0)
    kmeans.fit(df_pca)
    score = silhouette_score(df_pca, kmeans.labels_)
    silhouette_scores.append(score)

# Plot the silhouette scores to find the optimal number of clusters
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal Number of Clusters')
plt.show()


# Step 4: Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(df_pca)

# Step 5: Assign the cluster labels
df['Cluster'] = kmeans.labels_

# 2D Visualization using the first two principal components
def visualize_clusters_2d(df_pca, labels):
    plt.figure(figsize=(10, 7))
    plt.scatter(df_pca[:, 0], df_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2D PCA Clusters')
    plt.colorbar(label='Cluster Label')
    plt.show()

# 3D Visualization using the first three principal components
def visualize_clusters_3d(df_pca, labels):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df_pca[:, 0], df_pca[:, 1], df_pca[:, 2], c=labels, cmap='viridis', alpha=0.6)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.title('3D PCA Clusters')
    plt.colorbar(scatter, label='Cluster Label')
    plt.show()

# Call the functions
visualize_clusters_2d(df_pca, df['Cluster'])
visualize_clusters_3d(df_pca, df['Cluster'])





# Step 1: Compute the loadings matrix
loadings = final_pca.components_
loading_matrix = pd.DataFrame(loadings.T, columns=[f'PC{i+1}' for i in range(loadings.shape[0])])

# Step 2: Visualize the loadings
plt.figure(figsize=(10, 7))
sns.heatmap(loading_matrix, annot=True, cmap='coolwarm')
plt.title('PCA Loadings')
plt.xlabel('Principal Components')
plt.ylabel('Original Features')
plt.show()


# Step 1: Filter data points belonging to Cluster 4
cluster_4_data = df_pca[df['Cluster'] == 4]

# Step 2: Compute average loadings for PC1 and PC2 in Cluster 4
avg_loadings_cluster_4 = cluster_4_data.mean(axis=0)

# Step 3: Identify the more prevalent PC (higher average loading magnitude)
most_prevalent_pc = 'PC1' if np.abs(avg_loadings_cluster_4[0]) > np.abs(avg_loadings_cluster_4[1]) else 'PC2'

# Print the more prevalent PC in Cluster 4
#print(f"Cluster 4: Most Prevalent PC - {most_prevalent_pc}")

column_names = df.columns

# Specify the indices of the columns you want to print
indices_to_print = [3, 4, 5, 6]  # Adjust these indices as needed


# Print the names of the specified columns
for idx in indices_to_print:
    print(f"Column {idx}: {column_names[idx]}")





orig_df = df_health

#print(orig_df['Average Medicare Allowed Amount'])

# Select columns related to 'Provider Type'
provider_type_cols = 'Provider Type'

# Calculate mean and std for 'Average Submitted Charge Amount'
specialty_stats_submitted_chrg = orig_df.groupby(provider_type_cols)['Average Submitted Charge Amount'].agg(
    mean_submitted_chrg='mean', 
    std_submitted_chrg='std'
).reset_index()

# Calculate mean and std for 'Average Medicare Allowed Amount'
specialty_stats_allowed_amt = orig_df.groupby(provider_type_cols)['Average Medicare Allowed Amount'].agg(
    mean_allowed_amt='mean', 
    std_allowed_amt='std'
).reset_index()

# Calculate mean and std for 'Average Medicare Payment Amount'
specialty_stats_payment_amt = orig_df.groupby(provider_type_cols)['Average Medicare Payment Amount'].agg(
    mean_payment_amt='mean', 
    std_payment_amt='std'
).reset_index()

# Calculate mean and std for 'Average Submitted Charge Amount'
specialty_stats_standard_amt = orig_df.groupby(provider_type_cols)['Average Medicare Standardized Amount'].agg(
    mean_standard_amt='mean', 
    std_standard_amt='std'
).reset_index()

# Merge statistics for 'Average Submitted Charge Amount'
orig_df = orig_df.merge(specialty_stats_submitted_chrg, on=provider_type_cols, how='left')

# Merge statistics for 'Average Medicare Allowed Amount'
orig_df = orig_df.merge(specialty_stats_allowed_amt, on=provider_type_cols, how='left')

# Merge statistics for 'Average Medicare Payment Amount'
orig_df = orig_df.merge(specialty_stats_payment_amt, on=provider_type_cols, how='left')

# Merge statistics for 'Average Medicare Payment Amount'
orig_df = orig_df.merge(specialty_stats_standard_amt, on=provider_type_cols, how='left')

# Identify outliers for 'Average Submitted Charge Amount'
orig_df['Outlier_Submitted_Charge'] = (orig_df['Average Submitted Charge Amount'] > orig_df['mean_submitted_chrg'] + 3 * orig_df['std_submitted_chrg']) | \
                                     (orig_df['Average Submitted Charge Amount'] < orig_df['mean_submitted_chrg'] - 3 * orig_df['std_submitted_chrg'])


# Identify outliers for 'Average Medicare Allowed Amount'
orig_df['Outlier_Medicare_Allowed_Amount'] = (orig_df['Average Medicare Allowed Amount'] > orig_df['mean_allowed_amt'] + 3 * orig_df['std_allowed_amt']) | \
                                            (orig_df['Average Medicare Allowed Amount'] < orig_df['mean_allowed_amt'] - 3 * orig_df['std_allowed_amt'])

# Identify outliers for 'Average Medicare Payment Amount'
orig_df['Outlier_Medicare_Payment_Amount'] = (orig_df['Average Medicare Payment Amount'] > orig_df['mean_payment_amt'] + 3 * orig_df['std_payment_amt']) | \
                                            (orig_df['Average Medicare Payment Amount'] < orig_df['mean_payment_amt'] - 3 * orig_df['std_payment_amt'])

# Identify outliers for 'Average Medicare Payment Amount'
orig_df['Outlier_Medicare_Standardized_Amount'] = (orig_df['Average Medicare Standardized Amount'] > orig_df['mean_standard_amt'] + 3 * orig_df['std_standard_amt']) | \
                                            (orig_df['Average Medicare Standardized Amount'] < orig_df['mean_standard_amt'] - 3 * orig_df['std_standard_amt'])

# Scatter plot for 'Average Submitted Charge Amount'
plt.figure(figsize=(14, 8))
sns.scatterplot(x='Provider Type', y='Average Medicare Standardized Amount', hue='Outlier_Medicare_Standardized_Amount', data=orig_df, palette={False: 'blue', True: 'red'})
plt.title('Unusual Billing Patterns by Provider Type (Average Medicare Standardized Amount)')
plt.xlabel('Provider Type')
plt.ylabel('Average Medicare Standardized Amount')
plt.xticks(rotation=90)
plt.legend(title='Outlier')
plt.show()


# Scatter plot for 'Average Submitted Charge Amount'
plt.figure(figsize=(14, 8))
sns.scatterplot(x='Provider Type', y='Average Submitted Charge Amount', hue='Outlier_Submitted_Charge', data=orig_df, palette={False: 'blue', True: 'red'})
plt.title('Unusual Billing Patterns by Provider Type (Submitted Charge Amount)')
plt.xlabel('Provider Type')
plt.ylabel('Average Submitted Charge Amount')
plt.xticks(rotation=90)
plt.legend(title='Outlier')
plt.show()

# Scatter plot for 'Average Medicare Allowed Amount'
plt.figure(figsize=(14, 8))
sns.scatterplot(x='Provider Type', y='Average Medicare Allowed Amount', hue='Outlier_Medicare_Allowed_Amount', data=orig_df, palette={False: 'blue', True: 'red'})
plt.title('Unusual Billing Patterns by Provider Type (Medicare Allowed Amount)')
plt.xlabel('Provider Type')
plt.ylabel('Average Medicare Allowed Amount')
plt.xticks(rotation=90)
plt.legend(title='Outlier')
plt.show()

# Scatter plot for 'Average Medicare Payment Amount'
plt.figure(figsize=(14, 8))
sns.scatterplot(x='Provider Type', y='Average Medicare Payment Amount', hue='Outlier_Medicare_Payment_Amount', data=orig_df, palette={False: 'blue', True: 'red'})
plt.title('Unusual Billing Patterns by Provider Type (Medicare Payment Amount)')
plt.xlabel('Provider Type')
plt.ylabel('Average Medicare Payment Amount')
plt.xticks(rotation=90)
plt.legend(title='Outlier')
plt.show()


# Select the relevant columns for clustering
num_selected_columns = ['Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
                    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount']

df_selected = df_health[num_selected_columns]


num_pca = PCA()
num_X_pca = num_pca.fit_transform(df_selected)
num_explained_variance_ratio = num_pca.explained_variance_ratio_

# Calculate cumulative explained variance
num_cumulative_variance_ratio = np.cumsum(num_explained_variance_ratio)


# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(num_explained_variance_ratio, marker='o')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Principal Components')
plt.xticks(np.arange(len(num_explained_variance_ratio)), np.arange(1, len(num_explained_variance_ratio) + 1), rotation=90)
plt.grid(True)
plt.show()

# Plot cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(num_cumulative_variance_ratio, marker='o')
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio by Principal Components')
plt.xticks(np.arange(len(num_cumulative_variance_ratio)), np.arange(1, len(num_cumulative_variance_ratio) + 1), rotation=90)
plt.grid(True)
plt.show()


# Standardize numerical features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)
num_final_pca = PCA(n_components=2)
num_df_pca = num_final_pca.fit_transform(df_scaled)


# Calculate the silhouette scores for different numbers of clusters
silhouette_scores = []
for n in range(2, 11):  # Silhouette score is not defined for n=1
    kmeans = KMeans(n_clusters=n, random_state=0)
    kmeans.fit(num_df_pca)
    score = silhouette_score(num_df_pca, kmeans.labels_)
    silhouette_scores.append(score)

# Plot the silhouette scores to find the optimal number of clusters
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal Number of Clusters')
plt.show()




# Step 4: Apply K-Means clustering
num_kmeans = KMeans(n_clusters=3, random_state=0)
num_kmeans.fit(num_df_pca)

# Step 5: Assign the cluster labels
df_selected['Cluster'] = num_kmeans.labels_


#visualize_clusters_2d(num_df_pca, df_selected['Cluster'])






num_loadings = num_final_pca.components_
print(num_loadings)

num_loading_matrix = pd.DataFrame(num_loadings.T, columns=['PC1', 'PC2'], index=df_selected.columns[:-1])

print("PCA Loadings:")
print(num_loading_matrix)


# Visualize the loadings
plt.figure(figsize=(10, 7))
sns.heatmap(num_loading_matrix, annot=True, cmap='coolwarm')
plt.title('PCA Loadings')
plt.show()
'''

'''
# Provided PCA loadings and cluster centroids
num_centroids = np.array([[-1.41377357e-01, 1.20864640e-02],
                          [8.52326015e+01, 9.33707989e+00],
                          [9.41983736e+00, -1.04660053e+00]])

num_loadings = np.array([[0.51930559, 0.24845238],
                         [0.43875563, -0.89856242],
                         [0.51916396, 0.25060378],
                         [0.51796138, 0.26087304]])

# Transform centroids back to original feature space
centroids_original_space = np.dot(num_centroids, num_loadings.T)

# Create a DataFrame for better readability
centroids_df = pd.DataFrame(centroids_original_space, columns=['Average Medicare Allowed Amount',
                                                               'Average Submitted Charge Amount',
                                                               'Average Medicare Payment Amount',
                                                               'Average Medicare Standardized Amount'])

print("Cluster Centroids in Original Feature Space:")
print(centroids_df)


# Ensure indices are aligned before adding the provider type column
df_health.reset_index(drop=True, inplace=True)
df_selected.reset_index(drop=True, inplace=True)

df_selected['Provider Type'] = df_health['Provider Type']






# Analyze the distribution of provider types in each cluster
provider_type_distribution = df_selected.groupby('Cluster')['Provider Type'].value_counts().unstack().fillna(0)
#print("Provider Type Distribution in Each Cluster:")
#print(provider_type_distribution)

# Calculate and print proportions
provider_type_proportions = provider_type_distribution.div(provider_type_distribution.sum(axis=1), axis=0)
#print("Provider Type Proportions in Each Cluster:")
#print(provider_type_proportions)


provider_type_proportions_melted = provider_type_proportions.reset_index().melt(id_vars='Cluster', var_name='Provider Type', value_name='Proportion')
# Set the figure size
plt.figure(figsize=(14, 8))

# Create a bar plot for each cluster
sns.barplot(data=provider_type_proportions_melted, x='Provider Type', y='Proportion', hue='Cluster')

# Rotate x-axis labels for better readability
plt.xticks(rotation=90)
plt.title('Provider Type Proportions in Each Cluster')
plt.ylabel('Proportion')
plt.xlabel('Provider Type')
plt.legend(title='Cluster')
plt.show()
