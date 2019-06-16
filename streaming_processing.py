from pyspark import SparkContext
from pyspark.streaming import StreamingContext

sc = SparkContext('local[2]', 'NetworkWordCount')  # 2 threads
ssc = StreamingContext(sc, 1)  # 1, the interval is 1 second (for batches)
lines = ssc.socketTextStream('localhost', 9999)  # Input
words = lines.flatMap(lambda line: line.split(' '))  # List of words of each line
pairs = words.map(lambda word: (word, 1))
word_counts = pairs.reduceByKey(lambda num1, num2: num1 + num2)
word_counts.pprint()
ssc.start()
