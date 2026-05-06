from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Sample PySpark App") \
    .master("local[*]") \
    .getOrCreate()

data = [
    (1, "Alice", 34, "HR"),
    (2, "Bob", 45, "IT"),
    (3, "Cathy", 29, "Finance"),
    (4, "David", 40, "IT"),
    (5, "Eva", 38, "HR")
]

columns = ["id", "name", "age", "department"]

df = spark.createDataFrame(data, columns)

df.show()