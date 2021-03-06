
<a href="http://www.calstatela.edu/centers/hipic"><img align="left" src="https://avatars2.githubusercontent.com/u/4156894?v=3&s=100"><image/></a>
<img align="right" alt="California State University, Los Angeles" src="http://www.calstatela.edu/sites/default/files/groups/California%20State%20University%2C%20Los%20Angeles/master_logo_full_color_horizontal_centered.svg" style="width: 360px;"/>

#### Author: [Ruchi Singh](https://www.linkedin.com/in/ruchi-singh-68015945/)

#### Instructor: [Jongwook Woo](https://www.linkedin.com/in/jongwook-woo-7081a85)

#### Date: 05/20/2017

### Classification

### Prediction for FOOD related business
To have clearity in understanding which feature columns contributes in our prediction we can broadly categorize the business into different Categories like Food, Entertaiment, Medical, Services, Shooping, Education etc. For predicting the popularity of the Yelp business we decide to choose Food related business and feature columns.

Sub categories under Food category are 'Wine Bars','Vietnamese','vegetarian','vegan','Turkish','Thai','Tex-Mex','Tea Rooms','Tapas/Small Plates','Tapas Bars','Taiwanese','Szechuan','Sushi Bars','Steakhouses','Soup','Soul Food','Seafood','Sandwiches','Salad','Russian','Restaurants','restaurant' etc.

The feature columns related to food are review_count,stars,Take-out,GoodFor_lunch,GoodFor_dinner,GoodFor_breakfast,Noise_Level, Takes_Reservations,Delivery,Parking_lot,WheelchairAccessible,Alcohol,WaiterService,Wi-Fi.

## Download Data

download the "Business-Food.csv" file and upload in Databricks. Data-> default-> Create Table. Rename the table as "Food2" and check for all the columns datatype. 

This is the data to be used for training the machine learning algorithm.

## Logestic regression

The Logestic Regression classification model is used to predict the stars (popularity) for the business.The assumtion made here is that the business is unpopular if the Star is less than 3 and the business is popular if the Stars are more than 3.

### Prepare the Data
First, import the libraries you will need and prepare the training and test data:


```python
# Import Spark SQL and Spark ML libraries
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator

```

### Load Food table
Food table created is now loaded in Spark using SQL query:


```python
# Load the source data
csv = sqlContext.sql("Select * from food2")
```


```python
# Select features and label
# Logistic Regression
data = csv.select("review_count","Take-out", "GoodFor_lunch", "GoodFor_dinner", "GoodFor_breakfast","Noise_Level", "Takes_Reservations","Delivery","Parking_lot", "WheelchairAccessible","Alcohol", "WaiterService","Wi-Fi","stars")
```


```python
data.show(5)
```

### String Indexer
StringIndexer encodes a string column of labels to a column of label indices.


```python
def indexStringColumns(df, cols):
    #variable newdf will be updated several times
    newdata = df
    for c in cols:
        si = StringIndexer(inputCol=c, outputCol=c+"-x")
        sm = si.fit(newdata)
        newdata = sm.transform(newdata).drop(c)
        newdata = newdata.withColumnRenamed(c+"-x", c)
    return newdata

dfnumeric = indexStringColumns(data, ["Take-out","GoodFor_lunch", "GoodFor_dinner", "GoodFor_breakfast","Noise_Level", "Takes_Reservations","Delivery","Parking_lot", "WheelchairAccessible","Alcohol", "WaiterService","Wi-Fi"])


```


```python
dfnumeric.show(25)
```

### Encoder
One-hot encoding maps a column of label indices to a column of binary vectors, with at most a single one-value. This encoding allows algorithms which expect continuous features, in classification model, to use categorical features.


```python
def oneHotEncodeColumns(df, cols):
    from pyspark.ml.feature import OneHotEncoder
    newdf = df
    for c in cols:
        onehotenc = OneHotEncoder(inputCol=c, outputCol=c+"-onehot", dropLast=False)
        newdf = onehotenc.transform(newdf).drop(c)
        newdf = newdf.withColumnRenamed(c+"-onehot", c)
    return newdf

dfhot = oneHotEncodeColumns(dfnumeric, ["Take-out","GoodFor_lunch", "GoodFor_dinner", "GoodFor_breakfast","Noise_Level", "Takes_Reservations","Delivery","Parking_lot", "WheelchairAccessible","Alcohol", "WaiterService","Wi-Fi"])
```


```python
dfhot.show(25)
```

### Vector Assembler
VectorAssembler is a transformer that combines a given list of columns into a single vector column. It is useful for combining raw features and features generated by different feature transformers into a single feature vector, in order to train ML models like logistic regression.


```python
va = VectorAssembler(outputCol="features", inputCols=list(set(dfhot.columns)-set(['stars'])))
lpoints = va.transform(dfhot).select("features", "stars").withColumnRenamed("stars","label")
```

### Data Split
Split the data into training and test data in the ratio 80:20 using a random split.


```python
# Split the data
splits = lpoints.randomSplit([0.8, 0.2])
adulttrain = splits[0].cache()
adultvalid = splits[1].cache()
```

### Define the Pipeline
Now define a pipeline that creates a feature vector and trains a classification model


```python
lr = LogisticRegression(regParam=0.01, maxIter=1000, fitIntercept=True)
lrmodel = lr.fit(adulttrain)
lrmodel = lr.setParams(regParam=0.01, maxIter=500, fitIntercept=True).fit(adulttrain)
lrmodel.intercept

validpredicts = lrmodel.transform(adultvalid)
```


```python
validpredicts.show(5)
```

### Evaluate the model
Using a BinaryClassificationEvaluator the classification model used on the data is evaluated.


```python
from pyspark.ml.evaluation import BinaryClassificationEvaluator
bceval = BinaryClassificationEvaluator()
bceval.evaluate(validpredicts)
bceval.getMetricName()

bceval.setMetricName("areaUnderPR")
bceval.evaluate(validpredicts)
```


```python
display(validpredicts)
```

### Cross validation
It is is to ensure that every example from the original dataset has the same chance of appearing in the training and testing set.


```python
from pyspark.ml.tuning import CrossValidator
cv = CrossValidator().setEstimator(lr).setEvaluator(bceval).setNumFolds(2)
paramGrid = ParamGridBuilder().addGrid(lr.maxIter, [1000]).addGrid(lr.regParam, [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]).build()
cv.setEstimatorParamMaps(paramGrid)
cvmodel = cv.fit(adulttrain)

BinaryClassificationEvaluator().evaluate(cvmodel.bestModel.transform(adultvalid))
```

### Tune Parameters
You can tune parameters to find the best model for your data. A simple way to do this is to use  **TrainValidationSplit** to evaluate each combination of parameters defined in a **ParameterGrid** against a subset of the training data in order to find the best performing parameters.

#### Regularization 
It is a way of avoiding Imbalances in the way that the data is trained against the training data so that the model ends up being over fit to the training data. In other words It works really well with the training data but it doesn't generalize well with other data.
That we can use a **regularization parameter** to vary the way that the model balances that way.

#### Training ratio of 0.8
It is going to use 80% of the the data that it's got in its training set to train the model and then the remaining 20% is going to use to validate the trained model. 

In **ParamGridBuilder**, all possible combinations are generated from regParam, maxIter, threshold. So it is going to try each combination of the parameters with 80% of the the data to train the model and 20% to to validate it.


```python
# LogisticRegression with attribute 'threshold' in ParamGridBuilder and BinaryClassificationEvaluator
paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.3, 0.1, 0.01]).addGrid(lr.maxIter, [10, 5]).addGrid(lr.threshold, [0.35, 0.30]).build()

tvs = TrainValidationSplit(estimator=lr, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid, trainRatio=0.8)
model = tvs.fit(adulttrain)
```

### Test the Model
Now you're ready to apply the model to the test data.


```python
prediction = model.transform(adultvalid)
# LogisticRegression
predicted = prediction.select("features", "prediction", "probability", "label")

predicted.show(100)
```

### Compute Confusion Matrix Metrics: Only for Classification Logistic Regression not for Linear Regression
Classifiers are typically evaluated by creating a *confusion matrix*, which indicates the number of:
- True Positives
- True Negatives
- False Positives
- False Negatives

From these core measures, other evaluation metrics such as *precision* and *recall* can be calculated.

### Result
Precision (0.8762570727816253), Recall (0.7303376371612134): Precision becomes a little bit lower but the precision becomes much higher than previous no tuning example.


```python
# Only for Classification Logistic Regression 

tp = float(predicted.filter("prediction == 1.0 AND label == 1").count())
fp = float(predicted.filter("prediction == 1.0 AND label == 0").count())
tn = float(predicted.filter("prediction == 0.0 AND label == 0").count())
fn = float(predicted.filter("prediction == 0.0 AND label == 1").count())
metrics = spark.createDataFrame([
 ("TP", tp),
 ("FP", fp),
 ("TN", tn),
 ("FN", fn),
 ("Precision", tp / (tp + fp)),
 ("Recall", tp / (tp + fn))],["metric", "value"])
metrics.show()
```

### Review the Area Under ROC: Only for Classification Logistic Regression 
Another way to assess the performance of a classification model is to measure the area under a ROC curve for the model. the spark.ml library includes a **BinaryClassificationEvaluator** class that you can use to compute this.


```python
display(metrics)
```


```python
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")
aur = evaluator.evaluate(validpredicts)
print "AUR = ", aur
```


```python

```
References:
1. Microsoft DAT202.3x Implementing Predictive Analytics with Spark in Azure HDInsight 
1. Microsoft's DAT203x, Data Science and Machine Learning Essentials 
1. [URL of Data Source](https://s3.amazonaws.com/hipicdatasets/yelp_raw_fall_2016.csv)
1. [URL of Data Source2](https://www.yelp.com/dataset_challenge/dataset)
1. [URL of our  Github](https://github.com/rsingh26/DataScience/tree/master/MachineLearning)


