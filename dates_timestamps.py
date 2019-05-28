from pyspark.sql import SparkSession
from pyspark.sql.functions import dayofmonth, hour, dayofyear, month, weekofyear, format_number, date_format, year

spark = SparkSession.builder.appName('dates').getOrCreate()
df = spark.read.csv('data/Spark_DataFrames/appl_stock.csv', inferSchema=True, header=True)
dt = df.head(1)[0][0]
df.select(['Date', 'Open']).show()

# Use some functions
df.select(dayofmonth(df['Date'])).show()
df.select(month(df['Date'])).show()

# Average closing price per year
# df.select(year(df['Date'])).show()
# df.groupBy(year(df['Date'])).avg('Close').show()
newdf = df.withColumn('Year', year(df['Date']))
result = newdf.groupBy(newdf['Year']).mean().select('Year', 'avg(Close)')
result.orderBy('Year').withColumnRenamed('avg(Close)', 'Mean Close').select(['Year', format_number(
    'Mean Close', 2).alias('Avg Close')]).show()
