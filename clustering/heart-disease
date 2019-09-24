# Loading the data
heart_disease = read.csv("C:/Users/612024760/Downloads/project/Clustering Heart Disease Patient Data/datasets/heart_disease_patients.csv")

# Print the first ten rows of the data set
head(heart_disease, n = 10)

# Check that only numeric variables
lapply(heart_disease, class)

# Evidence that the data should be scaled?
summary(heart_disease)

# Remove id
heart_disease = heart_disease[ , !(names(heart_disease) %in% c("id"))]

# Scaling data and saving as a data frame
scaled = scale(heart_disease)

# What does data look like now?
summary(scaled)

# Set the seed so that results are reproducible
seed_val = 10
set.seed(seed_val, kind = "Mersenne-Twister", normal.kind = "Inversion")

# Select a number of clusters
k = 5

# Run the k-means algorithms
first_clust = kmeans(scaled, centers = k, nstart = 1)

# How many patients are in each group?
first_clust$size
# Set the seed
seed_val = 38
set.seed(seed_val, kind = "Mersenne-Twister", normal.kind = "Inversion")

# Run the k-means algorithms
k = 5
second_clust = kmeans(scaled, centers = k, nstart = 1)

# How many patients are in each group?
second_clust$size


# Adding cluster assignments to the data
heart_disease[,c("first_clust")] = first_clust$cluster
heart_disease[,c("second_clust")] = second_clust$cluster

# Load ggplot2
library(ggplot2)

# Creating the plots of age and chol for the first clustering algorithm
plot_one = ggplot(heart_disease, aes(x = age, y = chol, color = as.factor(first_clust))) + 
  geom_point()
plot_one 

# Creating the plots of age and chol for the second clustering algorithm
plot_two = ggplot(heart_disease, aes(x = age, y = chol, color = as.factor(second_clust))) + 
  geom_point()
plot_two


# Executing hierarchical clustering with complete linkage
hier_clust_1 = hclust(dist(scaled), method= "complete")

# Printing the dendrogram
plot(hier_clust_1)

# Getting cluster assignments based on number of selected clusters
hc_1_assign <- cutree(hier_clust_1, k = 5)


# Executing hierarchical clustering with single linkage
hier_clust_2 = hclust(dist(scaled), method = "single")

# Printing the dendrogram
plot(hier_clust_2)

# Getting cluster assignments based on number of selected clusters
hc_2_assign <- cutree(hier_clust_2, k = 5)

# Adding assignments of chosen hierarchical linkage
heart_disease['hc_clust'] = hc_1_assign

# Remove 'sex', 'first_clust', and 'second_clust' variables
hd_simple = heart_disease[, !(names(heart_disease) %in% c("sex", "first_clust", "second_clust"))]

# Getting mean and standard deviation summary statistics
clust_summary = do.call(data.frame, aggregate(. ~ hc_clust, data = hd_simple, function(x) c(avg = mean(x), sd = sd(x))))
clust_summary

# Plotting age and chol
plot_one = ggplot(heart_disease, aes(x = age, y = chol, color = as.factor(hc_clust))) + 
  geom_point()
plot_one 

# Plotting oldpeak and trestbps
plot_two = ggplot(heart_disease, aes(x = oldpeak, y = trestbps, color = as.factor(hc_clust))) + 
  geom_point()
plot_two

# Add TRUE if the algorithm shows promise, add FALSE if it does not
explore_kmeans = FALSE
explore_hierarch_complete = TRUE
explore_hierarch_single = FALSE
