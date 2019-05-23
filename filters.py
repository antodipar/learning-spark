from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('ops').getOrCreate()

df = spark.read.csv('data/Spark_DataFrames/appl_stock.csv', inferSchema=True, header=True)
df.head(5)

# Filter
df.filter('Close < 200').show()

# Filter and select with SQL notation
df.filter('Close < 200').select(['Open', 'Close']).show()
# The same as before but with more DataFrame notation
df.filter(df['Close'] < 200).select([df['Open'], df['Close']]).show()

# Several filters
df.filter((df['Close'] < 200) & ~(df['Open'] > 200)).show()
# Or
df.filter('Close < 200 and NOT Open > 200').show()

df2 = df.filter(df['Low'] == 197.16)
df2.printSchema()
result = df2.collect()  # List of Row objects
row = result[0]
row.asDict()