from pyspark.sql import SparkSession
from pyspark.sql.functions import format_number, col, max, min, corr, year, month

spark = SparkSession.builder.appName('exercise').getOrCreate()

# Read file and let spark infer the schema
df = spark.read.csv('data/Spark_DataFrames/walmart_stock.csv', inferSchema=True, header=True)

# What are the columns?
df.columns

# What does the schema look like?
df.printSchema()

# Print out the first five rows
df.head(5)

# Use "describe" to learn about the DataFrame
df.describe().show()
summary_df = df.describe()

# Format the number to show up two decimal places
formatted_summary_df = summary_df.select(
    col('summary'),
    format_number(col('Open').cast('float'), 2).alias('Open'),
    format_number(col('High').cast('float'), 2).alias('High'),
    format_number(col('Low').cast('float'), 2).alias('Low'),
    format_number(col('Close').cast('float'), 2).alias('Close'),
    format_number(col('Volume').cast('float'), 2).alias('Volume'),
    format_number(col('Adj Close').cast('float'), 2).alias('Adj Close')
)
formatted_summary_df.show()

# Create a new dataframe with a column called HV Ratio
newdf = df.select(col('High')/col('Volume'))
newdf.show()

# What day had the Peak High in Price?
high_max = df.groupBy().max('High').collect()[0][0]
day = df.filter(col('High') == high_max).collect()[0][0]
print(day)

# What is the mean of the Close column?
df.groupBy().avg('Close').show()

# What is the max and min of the Volume column?
df.select(max('Volume'), min('Volume')).show()

# How many days was the Close lower than 60 dollars?
df.filter(col('Close') < 60).count()

# What percentage of the time was the High greater than 80 dollars ?
df.filter(col('High') > 80).count() / df.count() * 100

#What is the Pearson correlation between High and Volume?
df.select(corr('High', 'Volume')).show()

# What is the max High per year?
df.groupBy(year(col('Date'))).max('High').show()

# What is the average Close for each Calendar Month?
avg_df = df.groupBy(month(col('Date')).alias('Month')).avg('Close').orderBy('Month')
avg_df.withColumnRenamed('avg(Close)', 'Mean Close').show()
