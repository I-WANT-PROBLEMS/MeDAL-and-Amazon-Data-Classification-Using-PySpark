import pyspark
from pyspark.sql import SparkSession, SQLContext
from pyspark import SparkConf, SparkContext
spark = SparkSession.builder.appName("Python Spark SQL basic example").config("spark.some.config.option", "some-value").getOrCreate()
df = spark.read.format("csv").option("header",True).option("delimiter", "\t").load("/home/user/spark-3.2.1-bin-hadoop3.2/bin/amazon.tsv")
df.show(5)
df = df.drop("marketplace")
df = df.drop("customer_id")
df = df.drop("review_id")
df = df.drop("product_id")
df = df.drop("product_parent")
df = df.drop("product_title")
df = df.drop("product_category")
df.show(5)
df = df.drop("helpful_votes")
df = df.drop("total_votes")
df = df.drop("vine")
df = df.drop("review_date")
df.show(5)
df.printSchema()
df.count()
df = df.filter(df.verified_purchase == "Y")
df.count()
import pyspark.sql.functions as F
df = df.withColumn("label", F.when((df.star_rating == "0") | (df.star_rating == "1") | (df.star_rating == "2"), "negative").when(df.star_rating == "3", "neutral").otherwise("positive"))
df.show(5)
df = df.drop("star_rating")
df = df.drop("verified_purchase")
df = df.withColumn("review", F.concat(df.review_headline, F.lit(' '), df.review_body))
df.show(5)
df = df.drop("review_headline")
df = df.drop("review_body")
df.show(5)
from pyspark.sql.functions import udf, col, lower, regexp_replace
from pyspark.ml.feature import Tokenizer
df = df.withColumn("review", lower(regexp_replace('review', "[^a-zA-Z0-9\\s]", "")))
# df.show(5)
from pyspark.sql.functions import split, col, length
df = df.withColumn("review", split(df.review,' '))
df.show(5)
from pyspark.ml.feature import CountVectorizer
countVec = CountVectorizer(inputCol = "review", outputCol = "review_vector")
model = countVec.fit(df)
newData = model.transform(df)
# newData.show(5)
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="label", outputCol="label_index")
indexed = indexer.fit(newData).transform(newData)
# indexed.show(5)
distinctData = indexed.distinct()
# distinctData.show(5)
train_data, test_data = distinctData.randomSplit([0.75,0.25])
# train_data.show()
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(labelCol="label_index", featuresCol="review",maxIter=100,regParam=0.01,elasticNetParam=0.1)
lr = lr.fit(train_data)
pred = lr.evaluate(test_data)
pred.accuracy