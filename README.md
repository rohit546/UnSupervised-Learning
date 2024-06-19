![image](https://github.com/rohit546/UnSupervised-Learning/assets/100420859/be5b5029-1ebf-4c02-8686-91d85ecc864e)# UnSupervised-Learning

Task1:
Unsupervised Learning: K-Means Clustering:
1. What does K-Means do?
K-means clustering is a type of unsupervised learning, which is used when you have unlabeled data (i.e.,
data without defined categories or groups). The goal of this algorithm is to find groups in the data, with the
number of groups represented by the variable K. The algorithm works iteratively to assign each data point
to one of K groups based on the features that are provided. Data points are clustered based on feature
similarity. The results of the K-means clustering algorithm are: The centroids of the K clusters, which can
be used to label new data Labels for the training data (each data point is assigned to a single cluster).

![image](https://github.com/rohit546/UnSupervised-Learning/assets/100420859/2433fc87-14e3-4318-9a13-98eec2a83c23)

2. Applications:
The K-means clustering algorithm is used to find groups which have not been explicitly labeled
in the data. This can be used to confirm business assumptions about what types of groups exist or
to identify unknown groups in complex data sets. Once the algorithm has been run and the groups
are defined, any new data can be easily assigned to the correct group. There are some applications
of k-means:
1. Banking (Fraud detection in credit card use/outlier detection, loyal vs churned customers)
2. Insurance Industry (Fraud detection in claims analysis, insurance risk of customers)
3. Publication/Media
4. Medicine, Biology
3. K-Means Algorithm:

![image](https://github.com/rohit546/UnSupervised-Learning/assets/100420859/0ed76ab4-3089-40db-882a-7240c8fd080b)

4. Implementation Steps:
STEP 1: Choose the number of clusters: k = 2

![image](https://github.com/rohit546/UnSupervised-Learning/assets/100420859/b4195d84-6286-4e54-b907-c0c3413def60)


![image](https://github.com/rohit546/UnSupervised-Learning/assets/100420859/9f9c2290-2377-4fc1-bb5f-64bbeb558eba)


![image](https://github.com/rohit546/UnSupervised-Learning/assets/100420859/114056e7-3a75-44f6-bbd8-44282829d8f6)


5. Working of K-Means Clustering
K-Means is an iterative algorithm and we have to repeat the steps (shown above) until the
algorithm converges. Each iteration, it will move centroids, calculated instances of points from
them, assign data points to nearest centroids (clusters). It will result in the clusters with minimum
error or the densest clusters.
K-Means Algorithm Working:
1. Randomly placing (as far as better) K centroids, one for each cluster.
2. Calculate the distance of each data point or object from the centroids (most popular
Euclidean distance or other measurement).
3. Assign each data point (object) to its closest centroid creating groups (clusters).
4. Recalculate the position of the K centroids (Mean of all points in the group becomes
centroid).
5. Repeat the steps 2-4, until the centroids no longer move.
It is a Heuristic Algorithm; means result may not be best possible outcome. So, it is common to
run the process multiple times with different starting conditions, means with randomized starting
centroids, it may give a better outcome. As this algorithm is usually very fast, running it multiple
times would not be a problem.
Choosing the right K:
As K-Means Clustering is an Unsupervised Algorithm, we do not have the ground truth (real
labels) available to test with like supervised algorithm.
But we can see how bad a cluster is; that is the average distance between the data points within a
cluster. Also, the average of the distances of data points from their cluster centroids can be used
as a metric of error for the clustering algorithm.
Number of Clusters or K, the Optimal K determining for a dataset is also a hard problem in K-Means clustering.
 We can use iteration using different values of K and plotting No. of K VS
Error/Accuracy. Then choose the K at elbow point (sharp shift) of error increase (As normally
increase in K will usually always decrease error).
Elbow Method:
The way to evaluate the choice of K is made using a parameter known as WCS. The WCSS
stands for within Cluster Sum of Squares. Here’s the formula representation for example:
when k = 3
Summation Distance (p, c) is the sum of distance of points in a cluster form the centroid

![image](https://github.com/rohit546/UnSupervised-Learning/assets/100420859/e5974ac1-f9da-4b8e-9db9-128858d6c95a)

The Elbow Method is then used to choose the best K value. In the depiction below we can see
that after 3 there’s no significant decrease in WCSS so 3 is the best here. Therefore, there’s an
elbow shape that forms and it is usually a good idea to pick the number where this elbow is
formed. There would be many times when the graph wouldn’t be this intuitive but with practice it
becomes easier.


![image](https://github.com/rohit546/UnSupervised-Learning/assets/100420859/ce57b264-7b6d-4fdc-baf5-4cb2ac26028b)

6. Python: K-Means Clustering:
Code is in the given Notebook.
Task 1:
Imagine that you have a house selling dataset, like house id, date, price, .... Lat, long, Size of
house (sqft). You can assign price to the house based on all the features and size of the house.
You need to apply house segmentation on this data based on the price and size of house. It is the
practice of partitioning a house into groups of individuals that have prices near to each other. It is
a significant strategy as a business can target these specific groups of houses by selling them to
their customers according to their requirements.
Use K Means clustering algorithm on the House Selling dataset. The goal of this task is to make
clusters of the houses based on their prices and size of house.
“House_Selling.csv” is attached.
Part 1:
Implement K-Means clustering without using any library functions
Apply the k-means algorithm and Euclidean distance to cluster the data-points into 3 clusters.
Note: you can get aid from “Tutorial exercises Clustering – K-means, Nearest Neighbor” at
University of Alberta,

https://webdocs.cs.ualberta.ca/~zaiane/courses/cmput695/F07/exercises/Exercises695Clus-
solution.pdf

or Getting-started-with-k-means-clustering-in-python, Dr Jesus Rogel-Salazar:
https://domino.ai/blog/author/jrogel,

Part 2:
Implement K-Means clustering using the Python library functions
a) Importing Libraries
a. Pandas
b. Numpy
c. Sklearn
d. seaborn or matplotlib (for visualization)
b) Importing the dataset
c) Using the elbow method to find the optimal number of clusters
d) Fitting K-Means to the dataset
e) Visualizing the clusters
NOTE: You may get the aid for implementation from following links
a. Sebastian Raschka, Yuxi (Hayden) Liu, and Vahid Mirjalili, “Working with Unlabeled Data –
Clustering Analysis,” in Machine Learning with PyTorch and Scikit-Learn, Packt Publishing,
2020, ch. 10, pp. 305 – 334. https://github.com/rasbt/machine-learning- book


b. https://developer.ibm.com/tutorials/awb-k-means-clustering-in-python/
c. k-means clustering: https://www.mathworks.com/help/stats/kmeans.html

Task 2:
K-Medoid:
Part 1: Implement K-Medoid clustering without using any library functions
Apply K-medoid clustering algorithm to cluster the data-points into five clusters, use the
Manhattan distance to find distance between data point and medoid.
Part 2: Implement K-Medoid clustering using the Python library functions
a) Importing Libraries
a. pandas
b. numpy
c. sklearn
d. seaborn or matplotlib (for visualization)
b) Importing the dataset
c) Using the elbow method to find the optimal number of clusters
d) Fitting K-Medoid to the dataset
e) Visualizing the clusters

NOTE: You may get the aid for implementation from following links
a. Sebastian Raschka, Yuxi (Hayden) Liu, and Vahid Mirjalili, “Working with Unlabeled Data –
Clustering Analysis,” in Machine Learning with PyTorch and Scikit-Learn, Packt Publishing,
2020, ch. 10, pp. 305 – 334. https://github.com/rasbt/machine-learning- book
b. https://developer.ibm.com/tutorials/awb-k-means-clustering-in-python/
c. k-means clustering: https://www.mathworks.com/help/stats/kmeans.html







