from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("TestJob").getOrCreate()
data = [("Trial", 1), ("Run", 2)]
df = spark.createDataFrame(data, ["name", "value"])
df.show()
spark.stop()
