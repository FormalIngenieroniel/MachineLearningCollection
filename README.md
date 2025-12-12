# 🤖 Machine Learning Project Collection, Supervised and Unsupervised Learning

This repository is a comprehensive collection of experiments and projects developed during a specialized Machine Learning course. It systematically covers fundamental concepts and advanced applications across both **Supervised Learning** and **Unsupervised Learning** paradigms.

The structure is divided into three main modules (C1, C2, and C3), progressing from basic prediction models to deep data evaluation and complex clustering tasks, serving as a robust professional portfolio.

---

## 📂 Project Structure and Learning Modules

### C1: Foundational Supervised Learning (Prediction Models)

* **Focus:** Introduction to building and evaluating basic supervised prediction models. This section establishes the fundamental workflow: data preparation, model training, and performance measurement.
* **Key Techniques Covered:**
    * **Regression Modeling:** Building models to estimate continuous target variables (e.g., estimating rental prices based on location and features).
    * **Classification:** Implementing and evaluating initial classification models.
    * **Model Comparison:** Early assessment of model performance and the risks of **overfitting** (e.g., comparing models on small datasets where 100% accuracy may be misleading).

### C2: Deep Data Context and Advanced Supervised Learning

This module intensifies the focus on the end-to-end Machine Learning pipeline, emphasizing data cleaning, feature engineering, and rigorous model evaluation in real-world scenarios.

* **Focus:** Developing robust predictive models for high-stakes business decisions (e.g., housing investment), with a strong emphasis on data quality, error analysis, and actionable business recommendations.
* **Key Techniques Covered:**
    * **Data Preprocessing:** Thorough data cleaning, format correction, handling of **Outliers** and **Missing Values**.
    * **Regression Algorithms:** Implementing and comparing advanced regression models, specifically leveraging **RandomForestRegressor** for its accuracy and resilience to extreme values.
    * **Feature Engineering & Selection:** Selection of the most relevant variables to simplify models and improve consistency (e.g., analyzing the correlation between stratum and administration price in housing data).
    * **Model Evaluation:** Detailed analysis of prediction errors (residuals), ensuring errors are distributed normally and without bias.
    * **Business Application:** Translating model results into clear, justified business recommendations (e.g., recommending a specific model for property price prediction for investment planning).

### C3: Unsupervised Learning (Clustering and Dimensionality)

This module shifts the focus to Unsupervised Learning, utilizing clustering methods to find hidden structures and patterns in complex, high-dimensional datasets, including image data. 

* **Focus:** Applying **K-Means Clustering** to various data distributions and complex problems like image grouping, and mastering the techniques for determining the optimal number of clusters.
* **Key Techniques Covered:**
    * **K-Means Algorithm:** Implementation and application of the algorithm on varied synthetic and real-world datasets.
    * **Optimal Cluster Determination:** Using both the **Elbow Method** (Elbow Curve) and the **Silhouette Coefficient** to rigorously determine the most appropriate number of groups ($k$) for a given dataset.
    * **High-Dimensional Data Analysis (Images):** Applying clustering to high-dimensional image data (e.g., face classification or general image grouping), including addressing the challenges of high-dimensionality (49152 dimensions).
    * **Cluster Interpretation:** Visualizing cluster **centroids** (average images) to understand the predominant characteristics of each group and interpreting the limitations of clustering in high-dimensional space.

---

## 🛠️ Key Technologies and Libraries

The projects are developed using Python and standard scientific computing libraries.

| Category | Libraries / Tools |
| :--- | :--- |
| **Language** | Python |
| **Data Manipulation** | `pandas`, `numpy` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Supervised Learning** | `scikit-learn` (sklearn), `RandomForestRegressor`, `KNeighborsClassifier`, `LogisticRegression` |
| **Unsupervised Learning** | `KMeans`, `PCA` (Dimensionality Reduction) |
| **Evaluation Metrics** | **MSE**, **RMSE**, Confusion Matrix, Accuracy, Precision, Recall, Elbow Method, Silhouette Coefficient |
