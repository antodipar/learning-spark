from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover, NGram
from pyspark.sql.functions import col, udf  # udf, user-defined function
from pyspark.sql.types import IntegerType

spark = SparkSession.builder.appName('nlp').getOrCreate()

sentence_df = spark.createDataFrame([
    (0, 'Hi I heard about Spark'),
    (1, 'I wish java could use case classes'),
    (2, 'Logistic,regression,models,are,neat')
], ['id', 'sentence'])
sentence_df.show()  # Simple dataframe

# Define tokenizers
tokenizer = Tokenizer(inputCol='sentence', outputCol='words')
regex_tokenizer = RegexTokenizer(inputCol='sentence', outputCol='words', pattern='\\W')  # Extract workd based on a pattern
# Define UDF
count_tokens = udf(lambda words: len(words), returnType=IntegerType())
# Apply them
tokenized = tokenizer.transform(sentence_df)
tokenized.show()
tokenized.withColumn('Number of Words (Tokens)', count_tokens(col('words'))).show()

rg_tokenized = regex_tokenizer.transform(sentence_df)
rg_tokenized.show()
rg_tokenized.withColumn('Number of Words (Tokens)', count_tokens(col('words'))).show()

# Use stop words remover to remove words that do not offer meaningful information
sentenceDataFrame = spark.createDataFrame([
    (0, ['I', 'saw', 'the', 'green', 'horse']),
    (1, ['Mary', 'had', 'a', 'little', 'lamb'])
], ['id', 'tokens'])
sentenceDataFrame.show()
remover = StopWordsRemover(inputCol='tokens', outputCol='filtered')  # Filter out common words
remover.transform(sentenceDataFrame).show()

# NGram
df = rg_tokenized.select(['id', 'words'])
ngram = NGram(n=2, inputCol='words', outputCol='grams')
ngram.transform(df).show(truncate=False)
