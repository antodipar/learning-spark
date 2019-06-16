from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer

spark = SparkSession.builder.appName('nlp').getOrCreate()
sentence_df = spark.createDataFrame([
    (0.0, 'Hi I heard about Spark'),
    (0.0, 'I wish java could use case classes'),
    (1.0, 'Logistic regression models are neat')
], ['label', 'sentence'])
sentence_df.show(truncate=False)

tokenizer = Tokenizer(inputCol='sentence', outputCol='words')
tokenized_df = tokenizer.transform(sentence_df)
tokenized_df.show(truncate=False)

hashing_tf = HashingTF(inputCol='words', outputCol='rawFeatures')
featurized_data = hashing_tf.transform(tokenized_df)
featurized_data.show(truncate=False)

idf = IDF(inputCol='rawFeatures', outputCol='features')
idf_model = idf.fit(featurized_data)
rescaled_data = idf_model.transform(featurized_data).select(['label', 'rawFeatures', 'features'])
rescaled_data.show(truncate=False)
# And now we feed rescaled_data dateframe into any machine learning =)

# What about CountVectorizer?
df = spark.createDataFrame([
    (0, 'a b c'.split(' ')),
    (1, 'a b b c a'.split(' '))
], ['id', 'words'])
df.show()

count_vectorizer = CountVectorizer(inputCol='words', outputCol='features', vocabSize=3, minDF=2.0)
model = count_vectorizer.fit(df)
result = model.transform(df)
result.show(truncate=False)
