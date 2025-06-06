# Task 8: Clustering with K-Means

##  Objective
Perform customer segmentation using K-Means clustering, an unsupervised learning algorithm, to group mall customers based on features such as age, annual income, and spending score.

##  Files Included
- `task8.py` – Python script containing the complete implementation of the task.
- `Mall_Customers.csv` – Dataset used for clustering.
- `task8-01.png` – Visualization of the Elbow Method.
- `task8-02.png` – Cluster visualization using PCA.
- `README.md` – This documentation file.

##  Tools & Libraries
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- PCA (for 2D visualization)

##  Key Steps
1. Loaded and scaled the dataset.
2. Applied PCA for optional 2D visualization.
3. Used the Elbow Method to find the optimal number of clusters (K).
4. Trained a K-Means model and assigned cluster labels.
5. Visualized the clusters in 2D using PCA.
6. Evaluated clustering quality using the Silhouette Score.

##  Result
- **Optimal number of clusters (K):** 5  
- **Silhouette Score:** ~0.55 (may vary slightly depending on run)

##  Dataset Source
Mall Customer Segmentation Dataset from [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

##  What I Learned
- How K-Means clustering works.
- How to evaluate clustering using inertia and silhouette score.
- The impact of normalization and dimensionality reduction on unsupervised learning.
