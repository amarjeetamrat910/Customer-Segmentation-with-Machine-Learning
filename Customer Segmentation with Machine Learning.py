#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")



# In[2]:


customer_data=pd.read_csv("Mall_Customers.csv")
customer_data



# # Data analysis 

# In[3]:


customer_data.head()


# In[4]:


customer_data.describe()


# In[5]:


customer_data.dtypes


# In[6]:


customer_data.isnull().sum()


# In[7]:


customer_data.drop(["CustomerID"],axis=1,inplace=True)
customer_data.head()


# In[8]:


max(customer_data["Age"])


# # Visualization of   dataset

# In[9]:


# Create a 2x2 subplot grid
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
plt.subplots_adjust(hspace=0.5, wspace=0.5)

# Plot each variable with smoothing in a separate subplot
sns.histplot(customer_data["Gender"], bins=2, kde=True, ax=axs[0, 0], kde_kws={'bw_adjust': 0.5})
axs[0, 0].set_title("Gender Distribution")

sns.histplot(customer_data["Age"], bins=30, kde=True, ax=axs[0, 1], kde_kws={'bw_adjust': 0.5})
axs[0, 1].set_title("Age Distribution")

sns.histplot(customer_data["Annual Income (k$)"], bins=30, kde=True, ax=axs[1, 0], kde_kws={'bw_adjust': 0.5})
axs[1, 0].set_title("Annual Income Distribution")

sns.histplot(customer_data["Spending Score (1-100)"], bins=20, kde=True, ax=axs[1, 1], kde_kws={'bw_adjust': 0.5})
axs[1, 1].set_title("Spending Score Distribution")

# Show the plots
plt.tight_layout()
plt.show()



# check who is purchasing more

# In[10]:


fig = plt.subplots(figsize=(5, 5))
sns.countplot(y="Gender",data=customer_data)
plt.show()


# In[11]:


fig, axs = plt.subplots(figsize=(10, 5), nrows=2, ncols=2)
plt.subplots_adjust(hspace=0.5, wspace=0.5)

# Plot violin plots
sns.violinplot(x="Gender", y="Age", data=customer_data, ax=axs[0, 0])
sns.violinplot(x="Gender", y="Annual Income (k$)", data=customer_data, ax=axs[0, 1])
sns.violinplot(x="Gender", y="Spending Score (1-100)", data=customer_data, ax=axs[1, 0])

# Hide the last subplot since there are only 3 variables
axs[1, 1].axis('off')

# Show the plot
plt.tight_layout()
plt.show()


# In[12]:


age_18_25 = customer_data["Age"][(customer_data['Age'] >= 18) & (customer_data['Age'] <= 25)]
age_26_35 = customer_data["Age"][(customer_data['Age'] >= 26) & (customer_data['Age'] <= 35)]
age_36_45 = customer_data["Age"][(customer_data['Age'] >= 36) & (customer_data['Age'] <= 45)]
age_46_55 = customer_data["Age"][(customer_data['Age'] >= 46) & (customer_data['Age'] <= 55)]
age_56_above = customer_data["Age"][(customer_data['Age'] >= 56)]

age_group_counts = [len(age_18_25), len(age_26_35), len(age_36_45), len(age_46_55), len(age_56_above)]
age_groups = ['18-25', '26-35', '36-45', '46-55', '56+']

plt.figure(figsize=(10, 6))
plt.bar(age_groups, age_group_counts, color='skyblue')
plt.xlabel('Age Group')
plt.ylabel('Number of Customers')
plt.title('Number of Customers in Different Age Groups')
plt.show()





# In[13]:


customer_data["Annual Income (k$)"]


# In[ ]:





# In[14]:


score_1_20 = customer_data[(customer_data['Spending Score (1-100)'] >= 1) & (customer_data['Spending Score (1-100)'] <= 20)]
score_21_40 = customer_data[(customer_data['Spending Score (1-100)'] >= 21) & (customer_data['Spending Score (1-100)'] <= 40)]
score_41_60 = customer_data[(customer_data['Spending Score (1-100)'] >= 41) & (customer_data['Spending Score (1-100)'] <= 60)]
score_61_80 = customer_data[(customer_data['Spending Score (1-100)'] >= 61) & (customer_data['Spending Score (1-100)'] <= 80)]
score_81_100 = customer_data[(customer_data['Spending Score (1-100)'] >= 81) & (customer_data['Spending Score (1-100)'] <= 100)]
x_axis=['1-20', '21-40', '41-60', '61-80', '81-100']
y_axis=[len(score_1_20), len(score_21_40), len(score_41_60), len(score_61_80), len(score_81_100)]
plt.figure(figsize=(10, 6))
plt.bar(x_axis, y_axis, color='skyblue')
plt.xlabel('score Group')
plt.ylabel('spending')
plt.title('spending vs income')
plt.show()


# In[ ]:





# In[15]:


annual_0_30 = customer_data[(customer_data['Annual Income (k$)'] >= 0) & (customer_data['Annual Income (k$)'] <= 30)]
annual_31_60 = customer_data[(customer_data['Annual Income (k$)'] > 31) & (customer_data['Annual Income (k$)'] <= 60)]
annual_61_90 = customer_data[(customer_data['Annual Income (k$)'] > 61) & (customer_data['Annual Income (k$)'] <= 90)]
annual_91_120 = customer_data[(customer_data['Annual Income (k$)'] > 91) & (customer_data['Annual Income (k$)'] <= 120)]
annual_121_150 = customer_data[(customer_data['Annual Income (k$)'] > 121) & (customer_data['Annual Income (k$)'] <= 150)]
x_axis = ['0-30', '31-60', '61-90', '91-120', '121-150']
y_axis = [
    len(annual_0_30),
    len(annual_31_60),
    len(annual_61_90),
    len(annual_91_120),
    len(annual_121_150)
]

plt.figure(figsize=(10, 6))
plt.bar(x_axis, y_axis, color='skyblue')
plt.xlabel('Income Range (k$)')
plt.ylabel('Number of Customers')
plt.title('Number of Customers vs Income Range')
plt.show()


# In[16]:


sns.scatterplot(x="Annual Income (k$)",y="Spending Score (1-100)",data=customer_data)


# In[17]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

X1 = customer_data.loc[:, ["Spending Score (1-100)", "Age"]]

wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(12, 6))
plt.grid()
plt.plot(range(1, 11), wcss, linewidth=2, color='red', marker='8')
plt.xlabel('K Value')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal K')
plt.show()


# In[18]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Assuming X is your clustering data and X1 is your plotting data
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans.fit(X1)

# Predicting clusters for the plotting data
labels = kmeans.predict(X1)

# Plotting the clusters
plt.figure(figsize=(10, 6))
plt.scatter(X1['Age'], X1['Spending Score (1-100)'], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='o', s=100, label='Centroids')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('KMeans Clustering')
plt.legend()
plt.show()




# In[19]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

X2 = customer_data.loc[:, ["Spending Score (1-100)", "Annual Income (k$)"]]

wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X2)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(12, 6))
plt.grid()
plt.plot(range(1, 11), wcss, linewidth=2, color='red', marker='8')
plt.xlabel('K Value')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal K')
plt.show()


# In[20]:


kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans.fit(X2)

# Predicting clusters for the plotting data
labels = kmeans.predict(X2)

# Plotting the clusters
plt.figure(figsize=(10, 6))
plt.scatter(X2['Spending Score (1-100)'],X2["Annual Income (k$)"], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='o', s=100, label='Centroids')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('KMeans Clustering')
plt.legend()
plt.show()


# In[34]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sample customer_data DataFrame
data = {
    'Age': np.random.randint(18, 70, 100),
    'Spending Score (1-100)': np.random.randint(1, 100, 100),
    'Annual Income (k$)': np.random.randint(20, 150, 100)
}
customer_data = pd.DataFrame(data)

# Create age groups
bins = [0, 30, 50, 100]
labels = ['Young', 'Middle-aged', 'Old']
customer_data['Age Group'] = pd.cut(customer_data['Age'], bins=bins, labels=labels)

# Define colors for each age group
colors = {'Young': 'r', 'Middle-aged': 'g', 'Old': 'b'}
color_column = customer_data['Age Group'].map(colors)

# Create 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(customer_data['Age'], customer_data['Spending Score (1-100)'], customer_data['Annual Income (k$)'],
           c=color_column, marker='o')

# Set labels and title
ax.set_xlabel('Age')
ax.set_ylabel('Spending Score (1-100)')
ax.set_zlabel('Annual Income (k$)')
plt.title('3D Scatter Plot with Different Colors for Age Groups')

# Create legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
                    for label, color in colors.items()]
plt.legend(handles=legend_elements, title='Age Group', bbox_to_anchor=(1, 0.5))

plt.show()


# In[ ]:




