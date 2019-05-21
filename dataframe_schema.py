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

# Sometimes we cannot leave Spark infer the Schema, we have to do it manually
data_schema = [StructField('age', IntegerType(), True),
               StructField('name', StringType(), True)]
final_struct = StructType(fields=data_schema)
df = spark.read.json('data/Spark_DataFrames/people.json', schema=final_struct)
df.printSchema()
