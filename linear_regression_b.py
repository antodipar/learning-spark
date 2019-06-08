from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import col

spark = SparkSession.builder.appName('lr_example').getOrCreate()
# Another way of load csv files
# We want to predict the Yearly Amount Spent variable
data = spark.read.format('csv'). \
    option('inferSchema', 'True'). \
    option('header', 'True'). \
    load('data/Linear_Regression/Ecommerce_Customers.csv')
data.show()
data.printSchema()

# We can see there are a total of 500 rows
data.describe().show()

# Let's prepare the dataframe for machine learning. Remember, we need a label and features (vector array)
# Let's focus first on numerical data
# VectorAssembler is a feature transformer that merges multiple columns into a vector column.

assembler = VectorAssembler(inputCols=['Avg Session Length', 'Time on App', 'Time on Website', 'Length of Membership'],
                            outputCol='features')
output = assembler.transform(data)
output.printSchema()
output.head(1)[0][-1]

final_data = output.select(col('features'), col('Yearly Amount Spent'))
final_data.show()

# Separate train data from test data
train_data, test_data = final_data.randomSplit([0.7, 0.3])
lr = LinearRegression(labelCol='Yearly Amount Spent')
model = lr.fit(train_data)
test_results = model.evaluate(test_data)
test_results.residuals.show()
print('Root Mean Squared Error')
print(test_results.rootMeanSquaredError)
print('r2')
print(test_results.r2)
test_results.predictions.show()

# Deploy the data
unlabeled_data = test_data.select(col('features'))
unlabeled_data.show()
predictions = model.transform(unlabeled_data)
