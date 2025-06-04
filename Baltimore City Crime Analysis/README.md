# cpsc479-p2
CPSC 479 Project 2: K-Means Clustering for Baltimore City Crime Data
# Group Members
  - Shayan Darian skdarian@csu.fullerton.edu
  - Jake Watson - jrwatson@csu.fullerton.edu
  - Yamato Eguchi - yamatoe1227@csu.fullerton.edu
# Project Description
Our algorithm starts by allocating memory on the host and device for the data, for the list of centroids, and for the sums of clusters.
<br>It then initializes the centroids by randomly selecting a coordinate from the given data.
<br>It then assigns each data point to the nearest centroid using squared Euclidian distance in parallel.
<br>The centroids are then recalculated as the mean of data points in each cluster.
<br>The process of assignment and centroid update is repeated until convergence or a predefined number of iterations.
<br>Convergence is then checked for, if true the loops breaks, if false we continue with the next iteration.
