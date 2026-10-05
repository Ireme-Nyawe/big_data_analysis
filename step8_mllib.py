from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

spark = SparkSession.builder \
    .appName("MLlibTutorial") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Load Iris dataset (built into sklearn, but we'll create it)
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
pdf = pd.DataFrame(iris.data, columns=iris.feature_names)
pdf["label"] = iris.target

df = spark.createDataFrame(pdf)

print("=== Iris Dataset ===")
df.show(5)
print(f"Total records: {df.count()}")

# Feature assembly
feature_cols = iris.feature_names
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Split data
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100)
pipeline_lr = Pipeline(stages=[assembler, lr])
model_lr = pipeline_lr.fit(train)

# Predictions
predictions_lr = model_lr.transform(test)
predictions_lr.select("features", "label", "prediction").show(10)

# Evaluate
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
accuracy_lr = evaluator.evaluate(predictions_lr)
print(f"Logistic Regression Accuracy: {accuracy_lr:.4f}")

# Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50)
pipeline_rf = Pipeline(stages=[assembler, rf])
model_rf = pipeline_rf.fit(train)
predictions_rf = model_rf.transform(test)
accuracy_rf = evaluator.evaluate(predictions_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")

spark.stop()