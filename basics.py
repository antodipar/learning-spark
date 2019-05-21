from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StringType, IntegerType, StructType

# Get or create a SparkSession object
spark = SparkSession.builder.appName('Basics').getOrCreate()
# Read json data format in to DataFrame
df = spark.read.json('data/Spark_DataFrames/people.json')
df.show()
df.printSchema()
df.columns
df.describe()  # Return a DataFrame[summary: string, age: string, name: string]
df.describe().show()

# # Sometimes we cannot leave Spark infer the Schema, we have to do it manually
# data_schema = [StructField('age', IntegerType(), True),
#                StructField('name', StringType(), True)]
# final_struct = StructType(fields=data_schema)
# df = spark.read.json('data/Spark_DataFrames/people.json', schema=final_struct)
# df.printSchema()

# Grab a column and return a DataFrame object
df_age = df.select('age')
# To select several columns
df.select(['name', 'age']).show()

# Grap the top two rows and return them in a list of Row objects
rows = df.head(2)

# Create new columns
newdf = df.withColumn('doubleage', df['age']*2)
newdf.show()

# Just to rename
df.withColumnRenamed('age', 'my_new_age').show()

# To interact using SQL
df.createOrReplaceTempView('people')
result = spark.sql('SELECT * from people')
new_results = spark.sql('SELECT name FROM people WHERE age=30')
