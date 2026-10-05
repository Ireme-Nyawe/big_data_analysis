"""
BT80733 — Big Data Analytics
Day 4 Exercise: PySpark Analytics on E-Commerce Dataset
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import (
    RegressionEvaluator,
    MulticlassClassificationEvaluator
)

# ============================================================
# INITIALISE SPARK
# ============================================================
spark = SparkSession.builder \
    .appName("Day4Exercise") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ============================================================
# PART A: DATA LOADING AND EXPLORATION
# ============================================================

# Q1: Load CSV with explicit schema
schema = StructType([
    StructField("order_id", StringType(), True),
    StructField("order_date", DateType(), True),
    StructField("customer_id", StringType(), True),
    StructField("category", StringType(), True),
    StructField("product", StringType(), True),
    StructField("unit_price", DoubleType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("discount", DoubleType(), True),
    StructField("total_amount", DoubleType(), True),
    StructField("region", StringType(), True),
    StructField("payment_method", StringType(), True),
    StructField("channel", StringType(), True),
    StructField("satisfaction", IntegerType(), True),
    StructField("delivery_days", IntegerType(), True),
])

df = spark.read.csv("ecommerce_sales.csv", header=True, schema=schema)

# Q2: Schema and first 10 rows
print("=" * 60)
print("Q2: SCHEMA AND SAMPLE DATA")
print("=" * 60)
df.printSchema()
df.show(10, truncate=False)

# Q3: Total orders
print("=" * 60)
print("Q3: TOTAL ORDERS")
print("=" * 60)
print(f"Total orders: {df.count()}")

# Q4: Unique customers
print("=" * 60)
print("Q4: UNIQUE CUSTOMERS")
print("=" * 60)
print(f"Unique customers: {df.select('customer_id').distinct().count()}")

# Q5: Date range
print("=" * 60)
print("Q5: DATE RANGE")
print("=" * 60)
df.agg(
    min("order_date").alias("Earliest"),
    max("order_date").alias("Latest")
).show()

# ============================================================
# PART B: DESCRIPTIVE ANALYTICS
# ============================================================

# Q6: Total revenue
print("=" * 60)
print("Q6: TOTAL REVENUE")
print("=" * 60)
df.agg(round(sum("total_amount"), 2).alias("TotalRevenue")).show()

# Q7: Average order value
print("=" * 60)
print("Q7: AVERAGE ORDER VALUE")
print("=" * 60)
df.agg(round(avg("total_amount"), 2).alias("AvgOrderValue")).show()

# Q8: Top 5 products by revenue
print("=" * 60)
print("Q8: TOP 5 PRODUCTS BY REVENUE")
print("=" * 60)
df.groupBy("product", "category") \
    .agg(round(sum("total_amount"), 2).alias("TotalRevenue")) \
    .orderBy(desc("TotalRevenue")) \
    .show(5)

# Q9: Revenue by category
print("=" * 60)
print("Q9: REVENUE BY CATEGORY")
print("=" * 60)
df.groupBy("category") \
    .agg(round(sum("total_amount"), 2).alias("TotalRevenue")) \
    .orderBy(desc("TotalRevenue")) \
    .show()

# Q10: Orders by region
print("=" * 60)
print("Q10: ORDERS BY REGION")
print("=" * 60)
df.groupBy("region") \
    .agg(
        count("*").alias("OrderCount"),
        round(sum("total_amount"), 2).alias("TotalRevenue")
    ) \
    .orderBy(desc("TotalRevenue")) \
    .show()

# ============================================================
# PART C: CUSTOMER ANALYTICS
# ============================================================

# Q11: Top 10 customers by spending
print("=" * 60)
print("Q11: TOP 10 CUSTOMERS BY SPENDING")
print("=" * 60)
df.groupBy("customer_id") \
    .agg(round(sum("total_amount"), 2).alias("TotalSpent")) \
    .orderBy(desc("TotalSpent")) \
    .show(10)

# Q12: Average satisfaction per category
print("=" * 60)
print("Q12: AVERAGE SATISFACTION PER CATEGORY")
print("=" * 60)
df.groupBy("category") \
    .agg(round(avg("satisfaction"), 2).alias("AvgSatisfaction")) \
    .orderBy(desc("AvgSatisfaction")) \
    .show()

# Q13: Payment methods
print("=" * 60)
print("Q13: PAYMENT METHOD ANALYSIS")
print("=" * 60)
df.groupBy("payment_method") \
    .agg(
        count("*").alias("OrderCount"),
        round(sum("total_amount"), 2).alias("TotalRevenue")
    ) \
    .orderBy(desc("OrderCount")) \
    .show()

# Q14: Average delivery time per region
print("=" * 60)
print("Q14: AVERAGE DELIVERY TIME PER REGION")
print("=" * 60)
df.groupBy("region") \
    .agg(round(avg("delivery_days"), 2).alias("AvgDeliveryDays")) \
    .orderBy("AvgDeliveryDays") \
    .show()

# Q15: Customers with > 5 orders
print("=" * 60)
print("Q15: CUSTOMERS WITH MORE THAN 5 ORDERS")
print("=" * 60)
frequent = df.groupBy("customer_id") \
    .agg(count("*").alias("OrderCount")) \
    .filter(col("OrderCount") > 5)
print(f"Number of frequent customers: {frequent.count()}")
frequent.orderBy(desc("OrderCount")).show(10)

# ============================================================
# PART D: ADVANCED ANALYTICS WITH SPARK SQL
# ============================================================

df.createOrReplaceTempView("sales")

# Q16: Monthly revenue trend
print("=" * 60)
print("Q16: MONTHLY REVENUE TREND")
print("=" * 60)
spark.sql("""
    SELECT
        DATE_FORMAT(order_date, 'yyyy-MM') AS Month,
        ROUND(SUM(total_amount), 2) AS TotalRevenue,
        COUNT(*) AS OrderCount
    FROM sales
    GROUP BY DATE_FORMAT(order_date, 'yyyy-MM')
    ORDER BY Month
""").show(12)

# Q17: Top 3 products per category
print("=" * 60)
print("Q17: TOP 3 PRODUCTS PER CATEGORY")
print("=" * 60)
spark.sql("""
    WITH product_revenue AS (
        SELECT category, product, SUM(total_amount) AS revenue
        FROM sales
        GROUP BY category, product
    ),
    ranked AS (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY category ORDER BY revenue DESC) AS rn
        FROM product_revenue
    )
    SELECT category, product, ROUND(revenue, 2) AS revenue
    FROM ranked
    WHERE rn <= 3
    ORDER BY category, revenue DESC
""").show(30)

# Q18: Correlation between discount and satisfaction
print("=" * 60)
print("Q18: CORRELATION — DISCOUNT vs SATISFACTION")
print("=" * 60)
corr_value = df.stat.corr("discount", "satisfaction")
print(f"Correlation(discount, satisfaction) = {corr_value:.4f}")

# Q19: Busiest day of the week
print("=" * 60)
print("Q19: BUSIEST DAY OF THE WEEK")
print("=" * 60)
spark.sql("""
    SELECT
        DATE_FORMAT(order_date, 'EEEE') AS DayOfWeek,
        COUNT(*) AS OrderCount
    FROM sales
    GROUP BY DATE_FORMAT(order_date, 'EEEE')
    ORDER BY OrderCount DESC
""").show()

# Q20: Average order value by channel
print("=" * 60)
print("Q20: AVERAGE ORDER VALUE BY CHANNEL")
print("=" * 60)
spark.sql("""
    SELECT
        channel,
        ROUND(AVG(total_amount), 2) AS AvgOrderValue,
        COUNT(*) AS OrderCount
    FROM sales
    GROUP BY channel
    ORDER BY AvgOrderValue DESC
""").show()

# ============================================================
# PART E: MACHINE LEARNING WITH MLLIB
# ============================================================

# --- Q21: Linear Regression — Predict total_amount ---
print("=" * 60)
print("Q21: LINEAR REGRESSION — PREDICT TOTAL AMOUNT")
print("=" * 60)

# Assemble features
assembler_lr = VectorAssembler(
    inputCols=["unit_price", "quantity", "discount"],
    outputCol="features"
)
lr_data = assembler_lr.transform(df).select("features", "total_amount")

# Split
train_lr, test_lr = lr_data.randomSplit([0.8, 0.2], seed=42)

# Train
lr = LinearRegression(featuresCol="features", labelCol="total_amount")
lr_model = lr.fit(train_lr)

# Predict
predictions_lr = lr_model.transform(test_lr)

# Evaluate
eval_rmse = RegressionEvaluator(
    labelCol="total_amount", predictionCol="prediction", metricName="rmse"
)
eval_r2 = RegressionEvaluator(
    labelCol="total_amount", predictionCol="prediction", metricName="r2"
)

rmse = eval_rmse.evaluate(predictions_lr)
r2 = eval_r2.evaluate(predictions_lr)

print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Coefficients: {lr_model.coefficients}")
print(f"Intercept: {lr_model.intercept:.4f}")

print("\nSample predictions:")
predictions_lr.select("features", "total_amount", "prediction").show(10)

# --- Q22: Random Forest Classifier — Predict satisfaction ---
print("=" * 60)
print("Q22: RANDOM FOREST — PREDICT SATISFACTION")
print("=" * 60)

assembler_rf = VectorAssembler(
    inputCols=["total_amount", "delivery_days", "discount", "quantity"],
    outputCol="features"
)
rf_data = assembler_rf.transform(df).select("features", "satisfaction")

train_rf, test_rf = rf_data.randomSplit([0.8, 0.2], seed=42)

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="satisfaction",
    numTrees=50,
    maxDepth=5,
    seed=42
)
rf_model = rf.fit(train_rf)
predictions_rf = rf_model.transform(test_rf)

eval_acc = MulticlassClassificationEvaluator(
    labelCol="satisfaction", predictionCol="prediction", metricName="accuracy"
)
accuracy = eval_acc.evaluate(predictions_rf)
print(f"Accuracy: {accuracy:.4f}")

print("\nConfusion Matrix:")
predictions_rf.groupBy("satisfaction", "prediction").count().orderBy("satisfaction", "prediction").show()

# --- Q23: K-Means Clustering — Customer Segmentation ---
print("=" * 60)
print("Q23: K-MEANS CLUSTERING — CUSTOMER SEGMENTATION")
print("=" * 60)

# Build customer-level features
customer_features = df.groupBy("customer_id").agg(
    sum("total_amount").alias("TotalSpent"),
    count("*").alias("OrderFrequency"),
    avg("satisfaction").alias("AvgSatisfaction")
)

assembler_km = VectorAssembler(
    inputCols=["TotalSpent", "OrderFrequency", "AvgSatisfaction"],
    outputCol="features"
)
km_data = assembler_km.transform(customer_features)

# Try different k values
for k in [2, 3, 4, 5]:
    kmeans = KMeans(featuresCol="features", k=k, seed=42)
    model = kmeans.fit(km_data)
    wssse = model.summary.trainingCost
    print(f"k={k}, WSSSE={wssse:.2f}")

# Use k=4 (adjust based on results)
kmeans = KMeans(featuresCol="features", k=4, seed=42)
km_model = kmeans.fit(km_data)
clustered = km_model.transform(km_data)

print("\nCluster distribution:")
clustered.groupBy("prediction").count().orderBy("prediction").show()

print("\nCluster centroids:")
for i, center in enumerate(km_model.clusterCenters()):
    print(f"Cluster {i}: TotalSpent={center[0]:.2f}, Frequency={center[1]:.2f}, Satisfaction={center[2]:.2f}")

# ============================================================
# CLEANUP
# ============================================================
spark.stop()
print("\n" + "=" * 60)
print("EXERCISE COMPLETE")
print("=" * 60)