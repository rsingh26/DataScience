
<a href="http://www.calstatela.edu/centers/hipic"><img align="left" src="https://avatars2.githubusercontent.com/u/4156894?v=3&s=100"><image/></a>
<img align="right" alt="California State University, Los Angeles" src="http://www.calstatela.edu/sites/default/files/groups/California%20State%20University%2C%20Los%20Angeles/master_logo_full_color_horizontal_centered.svg" style="width: 360px;"/>

#### Authors: [Ruchi Singh](https://www.linkedin.com/in/ruchi-singh-68015945/); [Kulwinder Kaur](https://www.linkedin.com/in/kulwinder-kaur-528a1b129/); [Ovanes Mikaelian](https://www.linkedin.com/in/hovik-mikaelian-93a257a3/)

#### Instructor: [Jongwook Woo](https://www.linkedin.com/in/jongwook-woo-7081a85)

#### Date: 05/20/2017

# CIS5560 Term Project Tutorial
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
Now load the review tip data into a DataFrame. This data consists of text that has been previously captured.


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
References:
1. Microsoft DAT202.3x Implementing Predictive Analytics with Spark in Azure HDInsight 
1. Microsoft's DAT203x, Data Science and Machine Learning Essentials 
1. [URL of Data Source](https://s3.amazonaws.com/hipicdatasets/yelp_raw_fall_2016.csv)
1. [URL of Data Source2](https://www.yelp.com/dataset_challenge/dataset)
1. [URL of our  Github](https://github.com/rsingh26/DataScience/tree/master/MachineLearning)
