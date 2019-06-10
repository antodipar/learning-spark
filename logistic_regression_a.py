from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

spark = SparkSession.builder.appName('log_regression').getOrCreate()
# Load data
my_data = spark.read.format('libsvm').load('data/Logistic_Regression/sample_libsvm_data.txt')

# Create instance and fitted
model = LogisticRegression(labelCol='label', featuresCol='features', predictionCol='prediction')
fitted_model = model.fit(my_data)

# Summary
summary = fitted_model.summary
predictions_df = summary.predictions
predictions_df.show()

# Let's split data by train and test dataset
train_data, test_data = my_data.randomSplit([0.7, 0.3])
# Retrain
final_model = LogisticRegression()
fitted_final_model = final_model.fit(train_data)
# Evaluate
prediction_and_labels = fitted_final_model.evaluate(test_data)
prediction_and_labels.predictions.show()

# Let's use evaluators
my_eval = BinaryClassificationEvaluator()
auc = my_eval.evaluate(prediction_and_labels.predictions)
print(f'AUC {auc}')

# Another way
auc = prediction_and_labels.areaUnderROC
