from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import col

spark = SparkSession.builder.appName('kmeans').getOrCreate()
data = spark.read.csv('data/Clustering/seeds_dataset.csv', inferSchema=True, header=True)
data.printSchema()
# By domain knowledge, we know that there three different types of seeds, that is, K=3

# Let's format the data
assembler = VectorAssembler(inputCols=data.columns, outputCol='features')
new_data = assembler.transform(data)

# Now scale the data
scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures')
final_data = scaler.fit(new_data).transform(new_data).select(col('scaledFeatures'))

# Train the model
model = KMeans(k=3, featuresCol='scaledFeatures')
fitted_model = model.fit(final_data)
wssse = fitted_model.computeCost(final_data)
print(f'WSSSE {wssse}')

centers = fitted_model.clusterCenters()
print(f'Centers {centers}')

preds = fitted_model.transform(final_data)
