import pyspark
from pyspark.sql import SparkSession, SQLContext
from pyspark import SparkConf, SparkContext
spark = SparkSession.builder.appName("Python Spark SQL basic example").config("spark.some.config.option", "some-value").getOrCreate()
df = spark.read.option("header",True).csv("full_data.csv")
# df = df.limit(20000000)
df.printSchema()
df = df.drop("_c0")
# df.show(5)
from pyspark.sql.functions import udf, col, lower, regexp_replace
from pyspark.ml.feature import Tokenizer
df = df.withColumn("Text", lower(regexp_replace('TEXT', "[^a-zA-Z0-9\\s]", "")))
# df.show(5)
from pyspark.sql.functions import split, col, length
df = df.withColumn("Text", split(df.Text,' '))
# df.show(5)
df2 = df.withColumn("Location", split(df.LOCATION,'\\|'))
# df2.show(5) 
df3 = df2.withColumn("Labels", split(df.LABEL,'\\|'))
# df3.show(5)
from pyspark.sql.functions import explode
import pyspark.sql.functions as F
from pyspark.sql.types import *
combine = F.udf(lambda x, y: list(zip(x, y)),
              ArrayType(StructType([StructField("locs", StringType()),
                                    StructField("labels", StringType())])))
df3 = df3.withColumn("new", combine("Location", "Labels")).withColumn("new", explode("new")).select("Text", col("new.locs").alias("Location"), col("new.labels").alias("Labels"))
# df3.show(5)
df3.Location = df3.Location.astype('int')
# df3.show(5)
df3 = df3.withColumn("Abbr", df3.Text[df3.Location])
# df3.show(5)
df3 = df3.drop("Location")
# df3.show(5)
df3 = df3.withColumn("Text + Abbr", F.concat(df3.Text, F.array(df3.Abbr)))
# df3.show(5)
from pyspark.ml.feature import HashingTF
hf = HashingTF(numFeatures = 1000, inputCol = "Text + Abbr", outputCol = "Model")
newData = hf.transform(df3)
# newData.show(5)
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="Labels", outputCol="LabelIndex")
indexed = indexer.fit(newData).transform(newData)
# indexed.show(5)
finalData = indexed.select('Model', 'LabelIndex')
# finalData.show(5)
distinctData = finalData.distinct()
# distinctData.show(5)
train_data, test_data = distinctData.randomSplit([0.75,0.25])
# train_data.show()
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(labelCol="LabelIndex", featuresCol="Model",maxIter=100,regParam=0.01,elasticNetParam=0.1)
lr = lr.fit(train_data)
pred = lr.evaluate(test_data)
pred.accuracy