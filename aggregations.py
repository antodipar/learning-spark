from pyspark.sql import SparkSession
from pyspark.sql.functions import countDistinct, avg, stddev, format_number

spark = SparkSession.builder.appName('aggs').getOrCreate()
df = spark.read.csv('data/Spark_DataFrames/sales_info.csv', inferSchema=True, header=True)
df.show()

# GroupBy + Aggregations
df.groupBy(df.Company).sum().show()
df.groupBy(df.Company).avg().show()
df.groupBy(df.Company).count().show()  # Number of rows
df.groupBy(df.Person).sum().show()

# Aggregations (across all rows)
df.agg({'Sales': 'sum'}).show()
df.agg({'Sales': 'avg'}).show()

# Other ways
group_data = df.groupBy('Company')
group_data.agg({'Sales': 'sum'}).show()  # Same as line 9

# Apply functions
df.select(countDistinct('Sales')).show()
df.select(avg('Sales')).show()  # Same as line 16
df.select(avg('Sales').alias('Average Sales')).show()  # Same as line 16
df.select(stddev('Sales')).show()

# Format the output
sales_std = df.select(stddev('Sales').alias('std'))
sales_std.show()
sales_std.select(format_number('std', 2).alias('Sales std')).show()

# Order
df.orderBy('Sales').show()
df.orderBy('Company').show()
df.orderBy(df['Sales'].desc())  # Descending order
