

# Customer Segmentation and Analysis

## Overview
This repository contains Python scripts and Jupyter notebooks for analyzing customer data from a mall. The analysis includes data preprocessing, exploratory data analysis (EDA), visualization, and customer segmentation using K-means clustering.

## Data Description
The dataset (`Mall_Customers.csv`) includes information on customers:
- **Gender**: Male or Female
- **Age**: Age of the customer
- **Annual Income (k$)**: Annual income of the customer in thousands of dollars
- **Spending Score (1-100)**: Spending score assigned by the mall based on customer behavior and spending nature

## Skills Demonstrated
- **Data Manipulation and Analysis**: Utilized NumPy and Pandas for data manipulation and statistical analysis.
- **Data Visualization**: Employed Matplotlib and Seaborn for creating informative visualizations.
- **Machine Learning**: Implemented K-means clustering for customer segmentation.

## Project Highlights

### Exploratory Data Analysis (EDA)
- Analyzed data distributions and relationships between variables.
- Visualized gender distribution, age distribution, annual income distribution, and spending score distribution.

### Customer Segmentation
- Identified optimal clusters using the elbow method to determine the best number of clusters.
- Visualized clusters based on age vs spending score and annual income vs spending score.

### Visualization Examples
- Created histograms, count plots, violin plots, and scatter plots to understand customer demographics and behavior.

## Project Accuracy
- Achieved accurate customer segmentation using K-means clustering with a clear distinction between different customer segments based on spending behavior and income.

## Code Snippets
### Example: Histogram and Count Plot
```python
# Gender distribution count plot
plt.figure(figsize=(8, 5))
sns.countplot(x='Gender', data=customer_data)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()
```

### Example: Customer Segmentation with K-means Clustering
```python
from sklearn.cluster import KMeans

# Selecting features for clustering
X = customer_data[['Age', 'Spending Score (1-100)']]

# K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# Visualizing clusters
plt.figure(figsize=(10, 6))
plt.scatter(X['Age'], X['Spending Score (1-100)'], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='o', s=100, label='Centroids')
plt.title('K-means Clustering: Age vs Spending Score')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
```

## Conclusion
This repository demonstrates proficiency in data analysis, visualization, and machine learning techniques for customer segmentation and analysis. Each script and notebook provides insights into customer behavior patterns and helps in making data-driven business decisions.

---
