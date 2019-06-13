from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

# from pyspark.ml.regression import DecisionTreeRegressor  # Also for regression

# Load data
spark = SparkSession.builder.appName('tree').getOrCreate()
data = spark.read.format('libsvm').load('data/Tree_Methods/sample_libsvm_data.txt')
data.show()
# Split data
train_data, test_data = data.randomSplit([0.7, 0.3])

# Instance models
dtc = DecisionTreeClassifier(featuresCol='features', labelCol='label')
rfc = RandomForestClassifier(featuresCol='features', labelCol='label', numTrees=100)
gbtc = GBTClassifier(featuresCol='features', labelCol='label')

# Fit classifier
fitted_dtc = dtc.fit(train_data)
fitted_rfc = rfc.fit(train_data)
fitted_gbtc = gbtc.fit(train_data)

# Predict test data
dtc_preds = fitted_dtc.transform(test_data)
dtc_preds.show()
rfc_preds = fitted_rfc.transform(test_data)
rfc_preds.show()
gbtc_preds = fitted_gbtc.transform(test_data)
gbtc_preds.show()

# Evaluate models
accuracy_eval = MulticlassClassificationEvaluator(metricName='accuracy')
print(f'Decision Tree Accuracy {accuracy_eval.evaluate(dtc_preds)}')
print(f'Random Forest Accuracy {accuracy_eval.evaluate(rfc_preds)}')
print(f'Gradient Boosted Trees Accuracy {accuracy_eval.evaluate(gbtc_preds)}')

# Feature importance
feature_importance = fitted_rfc.featureImportances
