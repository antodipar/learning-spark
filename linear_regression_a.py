from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import col

spark = SparkSession.builder.appName('lrex').getOrCreate()
# Load training dataset
training = spark.read.format('libsvm').load('data/Linear_Regression/sample_linear_regression_data.txt')
training.show()

# Create an instance of Linear Regression model
# Prediction parameter indicates the column name where predictions will be stored
lr = LinearRegression(featuresCol='features', labelCol='label', predictionCol='prediction')
lrModel = lr.fit(training)
print(f'Coefficients {lrModel.coefficients}')
print(f'Intercept {lrModel.intercept}')

# Print trained model summary
training_summary = lrModel.summary
print(f'r2 {training_summary.r2}')
print(f'Root mean squared error {training_summary.rootMeanSquaredError}')

# Let's separate training and test dataset
dataset = spark.read.format('libsvm').load('data/Linear_Regression/sample_linear_regression_data.txt')
train_data, test_data = dataset.randomSplit([0.7, 0.3])  # 70 % for training and 30 % for testing
train_data.describe().show()
test_data.describe().show()
model = lr.fit(train_data)
test_results = model.evaluate(test_data)  # Evaluate because we already know the true labels
test_results.residuals.show()

# Deploy the model
unlabeled_data = test_data.select(col('features'))
unlabeled_data.show()
predictions = model.transform(unlabeled_data)  # Transform because we don't know the true labels (apparently)
predictions.show()
