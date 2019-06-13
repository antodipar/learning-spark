from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

spark = SparkSession.builder.appName('tree').getOrCreate()
data = spark.read.csv('data/Tree_Methods/College.csv',inferSchema=True, header=True)

# Let's start formatting these data
assembler = VectorAssembler(inputCols=['Apps',
                                       'Accept',
                                       'Enroll',
                                       'Top10perc',
                                       'Top25perc',
                                       'F_Undergrad',
                                       'P_Undergrad',
                                       'Outstate',
                                       'Room_Board',
                                       'Books',
                                       'Personal',
                                       'PhD',
                                       'Terminal',
                                       'S_F_Ratio',
                                       'perc_alumni',
                                       'Expend',
                                       'Grad_Rate'],
                            outputCol='features')
output = assembler.transform(data)
indexer = StringIndexer(inputCol='Private', outputCol='PrivateIndex')
output_fixed = indexer.fit(output).transform(output)
output_fixed.show()
output_fixed.printSchema()
# You can also use a pipeline

final_data = output_fixed.select(['features', 'PrivateIndex'])
train_data, test_data = final_data.randomSplit([0.7, 0.3])

# Specify the models
dtc = DecisionTreeClassifier(featuresCol='features', labelCol='PrivateIndex')
rfc = RandomForestClassifier(featuresCol='features', labelCol='PrivateIndex')
gbtc = GBTClassifier(featuresCol='features', labelCol='PrivateIndex')

# Fit model using train data
fitted_dtc = dtc.fit(train_data)
fitted_rfc = rfc.fit(train_data)
fitted_gbtc = gbtc.fit(train_data)

# Transform test data
dtc_preds = fitted_dtc.transform(test_data)
rfc_preds = fitted_rfc.transform(test_data)
gbtc_preds = fitted_gbtc.transform(test_data)

# Evaluate models
binary_eval = BinaryClassificationEvaluator(labelCol='PrivateIndex')
dtc_auc = binary_eval.evaluate(dtc_preds)  # Note that dtc_preds dataframe contains everything we need
print(f'Decision Tree AUC: {dtc_auc}')
rfc_auc = binary_eval.evaluate(rfc_preds)
print(f'Random Forest AUC: {rfc_auc}')
gbtc_auc = binary_eval.evaluate(gbtc_preds)
print(f'Gradient Boosted Tree AUC: {gbtc_auc}')

# Let's tune the number of trees
rfc = RandomForestClassifier(featuresCol='features', labelCol='PrivateIndex', numTrees=150)
fitted_rfc = rfc.fit(train_data)
rfc_preds = fitted_rfc.transform(test_data)
rfc_auc = binary_eval.evaluate(rfc_preds)
print(f'Random Forest with 150 trees, AUC: {rfc_auc} ')

# What if we want to use other metrics for evaluation?
accuracy_eval = MulticlassClassificationEvaluator(labelCol='PrivateIndex', metricName='accuracy')
rfc_accuracy = accuracy_eval.evaluate(rfc_preds)
