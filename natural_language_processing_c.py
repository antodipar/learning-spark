from pyspark.sql import SparkSession
from pyspark.sql.functions import length, col, expr
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer, VectorAssembler
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName('nlp').getOrCreate()
data = spark.read.csv('data/Natural_Language_Processing/smsspamcollection/SMSSpamCollection',
                      inferSchema=True, sep='\t')
data.describe().show()
# Rename columns
data = data.withColumnRenamed('_c0', 'label').withColumnRenamed('_c1', 'text')
data.show()
# New column
data = data.withColumn('length', length(col('text')))
# Order by
data.groupBy('label').avg().show()
# Feature engineering
tokenizer = Tokenizer(inputCol='text', outputCol='token_text')
stop_remove = StopWordsRemover(inputCol='token_text', outputCol='stop_token')
count_vect = CountVectorizer(inputCol='stop_token', outputCol='c_vec')
idf = IDF(inputCol='c_vec', outputCol='tf_idf')
ham_spam_to_numeric = StringIndexer(inputCol='label', outputCol='indexLabel')
clean_up = VectorAssembler(inputCols=['length', 'tf_idf'], outputCol='features')
# Predicting model
model = NaiveBayes()
# Pipeline
data_prep_pipe = Pipeline(stages=[tokenizer, stop_remove, count_vect, idf, clean_up, ham_spam_to_numeric])
cleaner = data_prep_pipe.fit(data)
clean_data = cleaner.transform(data)
clean_data = clean_data.select([expr('indexLabel').alias('label'), 'features'])
clean_data.show()
# Train and test dataset
train_data, test_data = clean_data.randomSplit([0.7, 0.3])
fitted_model = model.fit(train_data)
preds = fitted_model.transform(test_data)
preds.show()
# Evaluate
eval = MulticlassClassificationEvaluator(metricName='accuracy')
accuracy = eval.evaluate(preds)
print(f'Accuracy {accuracy}')
