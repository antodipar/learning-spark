from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('log_regression').getOrCreate()
# Load data
df = spark.read.csv('data/Logistic_Regression/titanic.csv', inferSchema=True, header=True)
df.printSchema()
my_cols = df.select(['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
my_final_data = my_cols.na.drop()

# Working with categorical data
# SEX variable
# Use string indexer to convert any string to number
gender_indexer = StringIndexer(inputCol='Sex', outputCol='SexIndex')
# One-hot encoding
gender_encoder = OneHotEncoder(inputCol='SexIndex', outputCol='SexVec')

# EMBARKED variable
embark_indexer = StringIndexer(inputCol='Embarked', outputCol='EmbarkIndex')
embark_encoder = OneHotEncoder(inputCol='EmbarkIndex', outputCol='EmbarkVec')

assembler = VectorAssembler(inputCols=['Pclass', 'SexVec', 'EmbarkVec', 'Age', 'SibSp', 'Parch', 'Fare'],
                            outputCol='features')


model = LogisticRegression(featuresCol='features', labelCol='Survived')

# Create a pipeline (series of stages)
pipeline = Pipeline(stages=[gender_indexer, gender_encoder,
                            embark_indexer, embark_encoder,
                            assembler, model])

train_data, test_data = my_final_data.randomSplit([0.7, 0.3])
fitted_model = pipeline.fit(train_data)

# Evaluate on test dataset
results = fitted_model.transform(test_data)
results.show()
eval = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='Survived')
auc = eval.evaluate(results)
