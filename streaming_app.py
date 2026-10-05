from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

spark = SparkSession.builder \
    .appName("StreamingExample") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Read streaming data from socket
input_stream = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

# Parse the CSV data
parsed_stream = input_stream \
    .select(split(col("value"), ",").alias("data")) \
    .select(
        col("data")[0].alias("timestamp"),
        col("data")[1].alias("sensor"),
        col("data")[2].cast("double").alias("value")
    )

# Add event time
stream_with_time = parsed_stream \
    .withColumn("event_time", to_timestamp(col("timestamp")))

# Windowed aggregation with watermark
windowed_agg = stream_with_time \
    .withWatermark("event_time", "10 seconds") \
    .groupBy(
        window(col("event_time"), "10 seconds"),
        col("sensor")
    ) \
    .agg(
        avg("value").alias("avg_value"),
        max("value").alias("max_value"),
        min("value").alias("min_value"),
        count("*").alias("count")
    )

# Write to console
query = windowed_agg.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", "false") \
    .start()

query.awaitTermination()