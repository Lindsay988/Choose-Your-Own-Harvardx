# ==============================
# Gym Member Segmentation Project
# ==============================

# ------------------------------
# 1. Setup and Package Installation
# ------------------------------

# List of the required packages that are used in the project.
# This includes packages for data wrangling (tidyverse, dplyr), efficient data import (vroom),
# clustering (caret, cluster, mclust), visualization (ggplot2, plotly, factoextra), 
# and specific tools for dimensionality reduction (Rtsne) and date manipulation (lubridate). (Parvin, 2024)
packages <- c("tidyverse", "vroom", "ggplot2", "dplyr", "caret", "broom", 
              "data.table", "readr", "lubridate", "stringr", "plotly", "magrittr", 
              "modelr", "factoextra", "cluster", "mclust", "Rtsne")

# Any packages that are not installed, are installed.
if(!all(sapply(packages, require, character.only = TRUE))) {
  install.packages(packages[!sapply(packages, require, character.only = TRUE)])
}

# Required libraries are loaded.
lapply(packages, library, character.only = TRUE)

# Load the gym member dataset, that is placed in the GitHub repository.
dataset <- vroom::vroom("~/Choose-Your-Own-Harvardx/gym_members_exercise_tracking.csv")

# To understand the structure of the data, the first few rows of the dataset are displayed.
head(dataset)

# ------------------------------
# 2. Exploratory Data Analysis (EDA)
# ------------------------------

# 2.1 Basic Descriptive Statistics
# The variable distributions of the dataset is summarised, and any obvious issues is observed (e.g., outliers)

summary(dataset)

# 2.2 Data Distribution Visualization

#The distribution of each numerical variable is plotted into histograms, this is used for an assessment of the data provided. 

numeric_cols <- c("Age", "Weight (kg)", "Height (m)", "Max_BPM", "Avg_BPM", 
                  "Resting_BPM", "Session_Duration (hours)", "Calories_Burned", 
                  "Fat_Percentage", "Water_Intake (liters)", "BMI")
dataset %>% 
  select(all_of(numeric_cols)) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  geom_histogram(bins = 30) +
  facet_wrap(~key, scales = 'free') +
  ggtitle("Histograms of Numerical Variables") +
  theme_minimal()

# 2.3 Correlation Heatmap

# A heatmap is made to observe correlations among numerical variables to identify potential relationships and multicollinearity.

cor_matrix <- cor(dataset %>% select(where(is.numeric)), use = "complete.obs")
corrplot::corrplot(cor_matrix, method = "color", type = "upper", tl.col = "black", title = "Correlation Matrix")

# ------------------------------
# 3. Data Preprocessing
# ------------------------------

# 3.1 Handle Missing Data

#To avoid data loss, missing values in the numerical columns are replaced with the mean of each column.

dataset_clean <- dataset %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# 3.2 Normalise the numeric columns

# Numerical columns were scaled to standardise data for clustering, ensuring each feature has zero mean and unit variance
dataset_clean[numeric_cols] <- scale(dataset_clean[numeric_cols])

# ------------------------------
# 4. Clustering Models
# ------------------------------

# 4.1 The Elbow Method and Silhouette Score were used to determine the Optimal Number of Clusters. 

# The Elbow Method identified the ideal number of clusters by looking at the within-cluster sum of squares(WSS)
# The Silhouette Score provides more validation by assessing the quality of the cluster separation.

set.seed(123)  # Ensures reproducibility
wss <- factoextra::fviz_nbclust(dataset_clean[numeric_cols], kmeans, method = "wss") +
  ggtitle("Elbow Method for Finding Optimal Clusters") +
  theme_minimal()

silhouette_score <- factoextra::fviz_nbclust(dataset_clean[numeric_cols], kmeans, method = "silhouette") +
  ggtitle("Silhouette Score for Finding Optimal Clusters") +
  theme_minimal()

# The Elbow Method and Silhouette Score plots are both displayed.
print(wss)
print(silhouette_score)

# 4.2 Apply K-Means and Hierarchical Clustering for Comparison

# According to the Elbow and Silhouette methods k=3 was chosen to do K-Means Clustering.

k <- 3  
kmeans_result <- kmeans(dataset_clean[numeric_cols], centers = k, nstart = 25)

# To compare clustering techniques, Hierarchical Clustering was applied. 

hierarchical_result <- hclust(dist(dataset_clean[numeric_cols]), method = "ward.D2")
cluster_assignment_hc <- cutree(hierarchical_result, k)

# Cluster assignments were added to the dataset for both the K-Means Clustering and the the Hierarchical Clustering. 

dataset_clean$KMeans_Cluster <- as.factor(kmeans_result$cluster)
dataset_clean$Hierarchical_Cluster <- as.factor(cluster_assignment_hc)

# ------------------------------
# 5. Advanced Visualization of Clusters
# ------------------------------

# 5.1 Principal Component Analysis (PCA)

# PCA reduces dimensionality, showing clusters along principal components (with the most variance).
pca_result <- prcomp(dataset_clean[numeric_cols], scale. = TRUE)
factoextra::fviz_pca_ind(pca_result, geom.ind = "point", habillage = as.factor(kmeans_results$cluster), 
                         addEllipses = TRUE, palette = "jco") +
  ggtitle("PCA - KMeans Clusters") +
  theme_minimal()

# 5.2 t-SNE for Clustering Visualization

# t-SNE provides a nonlinear dimensionality reduction, ideal for visualising high-dimensional clusters in 2D.

tsne_result <- Rtsne::Rtsne(as.matrix(dataset_clean[numeric_cols]), dims = 2, perplexity = 30)
dataset_clean$TSNE1 <- tsne_result$Y[,1]
dataset_clean$TSNE2 <- tsne_result$Y[,2]
ggplot(dataset_clean, aes(x = TSNE1, y = TSNE2, color = KMeans_Cluster)) +
  geom_point() +
  ggtitle("t-SNE - KMeans Clusters") +
  theme_minimal()

# ------------------------------
# 6. Cluster Interpretation and Analysis
# ------------------------------

# 6.1 Cluster Analysis with Descriptive Statistics for Each Cluster.

#To interpret the common traits within the clusters each cluster’s characteristics were summarised. 

kmeans_summary <- dataset_clean %>%
  group_by(KMeans_Cluster) %>%
  summarise(
    Avg_Age = mean(Age, na.rm = TRUE),
    Avg_Weight = mean(`Weight (kg)`, na.rm = TRUE),
    Avg_Height = mean(`Height (m)`, na.rm = TRUE),
    Avg_Max_BPM = mean(Max_BPM, na.rm = TRUE),
    Avg_Session_Duration = mean(`Session_Duration (hours)`, na.rm = TRUE),
    Avg_Calories_Burned = mean(Calories_Burned, na.rm = TRUE),
    Avg_Fat_Percentage = mean(Fat_Percentage, na.rm = TRUE),
    Avg_BMI = mean(BMI, na.rm = TRUE)
  )

print(kmeans_summary)

# ------------------------------
# 7. Cluster Evaluation with Silhouette Scores
# ------------------------------

#Silhouette scores were used to evaluate the clustering performance of both the K-Means and Hierarchical clustering. 

silhouette_kmeans <- cluster::silhouette(kmeans_result$cluster, dist(dataset_clean[numeric_cols]))
silhouette_hierarchical <- cluster::silhouette(cluster_assignment_hc, dist(dataset_clean[numeric_cols]))

# Silhouette plot for K-Means Clustering
factoextra::fviz_silhouette(silhouette_kmeans) + 
  ggtitle("Silhouette Plot for KMeans Clustering") +
  theme_minimal()

# Silhouette plot for Hierarchical Clustering
factoextra::fviz_silhouette(silhouette_hierarchical) + 
  ggtitle("Silhouette Plot for Hierarchical Clustering") +
  theme_minimal()

# ------------------------------
# 8. Tailor Gym Programs Based on Cluster Characteristics
# ------------------------------

# Function to assign recommended gym programs according to the cluster traits

assign_gym_program <- function(cluster) {
  if (cluster == 1) {
    return("High-Intensity Interval Training (HIIT) - Short and Intense")
  } else if (cluster == 2) {
    return("Endurance and Strength Training - Long Duration Workouts")
  } else if (cluster == 3) {
    return("Beginner-Friendly Cardio and Strength - Moderate Intensity")
  } else {
    return("General Fitness Program")
  }
}

#Tailored gym programs were assigned to each member based on their cluster.

dataset_clean$Recommended_Program <- sapply(dataset_clean$KMeans_Cluster, assign_gym_program)

# 8.1 Summary of Recommended Programs per Cluster

# A count of members assigned to each program type across clusters is displayed.

program_summary <- dataset_clean %>%
  group_by(Recommended_Program) %>%
  summarise(Members_Count = n())

print("Summary of Gym Programs Assigned to Clusters:")
print(program_summary)

# 8.2 Visualisation of Gym Program Distribution Across Clusters
# Bar plot to show the distribution of recommended programs per cluster.
ggplot(dataset_clean, aes(x = KMeans_Cluster, fill = Recommended_Program)) +
  geom_bar() +
  ggtitle("Distribution of Gym Programs Across KMeans Clusters") +
  xlab("KMeans Cluster") +
  ylab("Number of Members") +
  theme_minimal()


