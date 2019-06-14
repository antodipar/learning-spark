from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.clustering import KMeans


spark = SparkSession.builder.appName('clustering').getOrCreate()

data = spark.read.format('libsvm').load('data/Clustering/sample_kmeans_data.txt')
final_data = data.select(col('Features').alias('features'))
final_data.show()

# Create model
model = KMeans().setK(3).setSeed(1)
fitted_model = model.fit(final_data)
wssse = fitted_model.computeCost(final_data)  # sum of squared distances
print(wssse)

# Get the centers
centers = fitted_model.clusterCenters()

# What cluster each point belongs to?
results = fitted_model.transform(final_data)
results.show()
