
## Text Analysis
To create a classification model that analyse tip text to predict the likes.
### Import Spark SQL and Spark ML Libraries

First, import the libraries you will need:


```python
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer, StopWordsRemover
```

### Load Source Data
Now load the tweets data into a DataFrame. This data consists of tweets that have been previously captured and classified as positive or negative.


```python
text_csv = sqlContext.sql("Select * from tipcleaned")

text_csv.show(5)
```


```python
display(text_csv.groupBy("likes").count().orderBy("likes"))
```

### Prepare the Data
The features for the classification model will be derived from the tip text. The label is the like (between 1-10)


```python
textdata = text_csv.select("text", col("likes").cast("Int").alias("label"))
textdata.show(truncate = False)
```

### Split the Data
In common with most classification modeling processes, you'll split the data into a set for training, and a set for testing the trained model.


```python
splits = textdata.randomSplit([0.7, 0.3],seed=0)
textrain = splits[0]
textest = splits[1].withColumnRenamed("label", "trueLabel")
textrain_rows = textrain.count()
textest_rows = textest.count()
print "Training Rows:", textrain_rows, " Testing Rows:", textest_rows
```


```python
textest.show(5)
```


```python
textdata.show(5,truncate = False)
```

### Define the Pipeline
The pipeline for the model consist of the following stages:
- A Tokenizer to split the tweets into individual words.
- A StopWordsRemover to remove common words such as "a" or "the" that have little predictive value.
- A HashingTF class to generate numeric vectors from the text values.
- A LogisticRegression algorithm to train a binary classification model.


```python
# convert sentence to words' list
tokenizer = Tokenizer(inputCol="text", outputCol="Words")
# remove stop words
swr = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="MeaningfulWords")
# convert word to number as word frequency
hashTF = HashingTF(inputCol=swr.getOutputCol(), outputCol="features")
# set the model
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10, regParam=0.01)

# process pipeline with the series of transforms - 4 transforms
pipeline = Pipeline(stages=[tokenizer, swr, hashTF, lr])
```

### Run the Pipeline as an Estimator
The pipeline itself is an estimator, and so it has a **fit** method that you can call to run the pipeline on a specified DataFrame. In this case, you will run the pipeline on the training data to train a model.


```python
piplineModel = pipeline.fit(textrain)
print "Pipeline complete!"
```

### Test the Pipeline Model
The model produced by the pipeline is a transformer that will apply all of the stages in the pipeline to a specified DataFrame and apply the trained model to generate predictions. In this case, you will transform the **test** DataFrame using the pipeline to generate label predictions.


```python
prediction = piplineModel.transform(textest)
predicted = prediction.select("text", "prediction", "trueLabel")
predicted.show(10)
```


```python
predicted10 = prediction.select("*")
predicted10.show(10)
```

### Compute Confusion Matrix Metrics
Classifiers are typically evaluated by creating a *confusion matrix*, which indicates the number of:
- True Positives
- True Negatives
- False Positives
- False Negatives

From these core measures, other evaluation metrics such as *precision* and *recall* can be calculated.


```python
tp = float(predicted10.filter("prediction == 1.0 AND truelabel == 1").count())
fp = float(predicted10.filter("prediction == 1.0 AND truelabel == 0").count())
tn = float(predicted10.filter("prediction == 0.0 AND truelabel == 0").count())
fn = float(predicted10.filter("prediction == 0.0 AND truelabel == 1").count())
metrics = spark.createDataFrame([
 ("TP", tp),
 ("FP", fp),
 ("TN", tn),
 ("FN", fn),
 ("Precision", tp / (tp + fp)),
 ("Recall", tp / (tp + fn))],["metric", "value"])
metrics.show()
```

### Review the Area Under ROC
Another way to assess the performance of a classification model is to measure the area under a ROC curve for the model. the spark.ml library includes a **BinaryClassificationEvaluator** class that you can use to compute this.


```python
from pyspark.ml.evaluation import BinaryClassificationEvaluator
# LogisticRegression: rawPredictionCol="prediction", metricName="areaUnderROC"
evaluator = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="prediction", metricName="areaUnderROC")
aur = evaluator.evaluate(prediction)
print "AUR = ", aur

```
