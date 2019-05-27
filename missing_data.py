from pyspark.sql import SparkSession
from pyspark.sql.functions import mean

spark = SparkSession.builder.appName('miss').getOrCreate()
df = spark.read.csv('data/Spark_DataFrames/ContainsNull.csv', inferSchema=True, header=True)
df.show()

# Drop rows with any number of null values
df.na.drop().show()
# Instead, drop rows containing at least two null values
df.na.drop(thresh=2).show()
# Other ways to drop null values. Remove those rows with any number of null values
df.na.drop(how='any').show()
# Remove those rows where all values are null
df.na.drop(how='all').show()
# We can also use subsets to remove rows where the subset is null
df.na.drop(subset=['Sales']).show()

# What if we don't want to drop missing values and fill them up instead
# Fill columns of string type with "FILL ME"
df.na.fill('FILL ME').show()
# Fill columns of numeric type with 1234
df.na.fill(1234).show()
# To specify a certain column
df.na.fill('No name', subset=['Name']).show()
df.na.fill(1234567, subset=['Sales']).show()

# To fill missing values with the mean for the column
mean_value = df.select(mean(df['Sales']).alias('mean')).collect()
mean_sales = mean_value[0].asDict()['mean']
# Other way
mean_sales = mean_value[0][0]  # Grab the first row of the list and then the first value of the row
df.na.fill(mean_sales, subset=['Sales']).show()
# Even we can do this in one line
df.na.fill(df.select(mean(df['Sales'])).collect()[0][0], subset=['Sales']).show()  # Better do not do this
