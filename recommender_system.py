# The two main types of recommender systems are Content-Based and Collaborative Filtering (CF)
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALS  # Alternating least squares
from pyspark.ml.evaluation import RegressionEvaluator


spark = SparkSession.builder.appName('recommender').getOrCreate()
data = spark.read.csv('data/Recommender_Systems/movielens_ratings.csv', inferSchema=True, header=True)
train_data, test_data = data.randomSplit([0.8, 0.2])
model = ALS(maxIter=5, regParam=0.01, userCol='userId', itemCol='movieId', ratingCol='rating')
fitted_model = model.fit(train_data)
preds = fitted_model.transform(test_data)
preds.show()
# Evaluate the model
evaluator = RegressionEvaluator(labelCol='rating', metricName='rmse', predictionCol='prediction')
rmse = evaluator.evaluate(preds)
print(f'Root mean square error {rmse}')

# Emulate single user
single_user = test_data.filter(col('userId') == 11).select(['movieId', 'userId'])
single_user.show()
recommendations = fitted_model.transform(single_user)
recommendations.orderBy(col('prediction'), ascending=False).show()