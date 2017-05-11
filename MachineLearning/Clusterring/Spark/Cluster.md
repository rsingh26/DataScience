
## Clustering
In this exercise, you will use K-Means clustering to segment customer data into five clusters.

### Import the Libraries
You will use the **KMeans** class to create your model. This will require a vector of features, so you will also use the **VectorAssembler** class.


```python
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
```

### Load Source Data
The source data for your clusters is in a comma-separated values (CSV) file, and incldues the following features:
- CustomerName: The custome's name
- Age: The customer's age in years
- MaritalStatus: The custtomer's marital status (1=Married, 0 = Unmarried)
- IncomeRange: The top-level for the customer's income range (for example, a value of 25,000 means the customer earns up to 25,000)
- Gender: A numeric value indicating gender (1 = female, 2 = male)
- TotalChildren: The total number of children the customer has
- ChildrenAtHome: The number of children the customer has living at home.
- Education: A numeric value indicating the highest level of education the customer has attained (1=Started High School to 5=Post-Graduate Degree
- Occupation: A numeric value indicating the type of occupation of the customer (0=Unskilled manual work to 5=Professional)
- HomeOwner: A numeric code to indicate home-ownership (1 - home owner, 0 = not a home owner)
- Cars: The number of cars owned by the customer.


```python
# Adopt shcmea to read csv data set in the schema. 

'''customers= spark.read.csv('swift://Unsupervised13.' + name + '/customers.csv', header="true", inferSchema="true",mode="DROPMALFORMED")'''
csv = sqlContext.sql("Select * from food2")


```


```python
data = csv.select("review_count","Take-out", "GoodFor_lunch", "GoodFor_dinner", "GoodFor_breakfast","stars")
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

dfnumeric = indexStringColumns(data, ["Take-out","GoodFor_lunch", "GoodFor_dinner", "GoodFor_breakfast"])
```

### Hot Encoder

One-hot encoding maps a column of label indices to a column of binary vectors, with at most a single one-value.


```python
def oneHotEncodeColumns(df, cols):
    from pyspark.ml.feature import OneHotEncoder
    newdf = df
    for c in cols:
        onehotenc = OneHotEncoder(inputCol=c, outputCol=c+"-onehot", dropLast=False)
        newdf = onehotenc.transform(newdf).drop(c)
        newdf = newdf.withColumnRenamed(c+"-onehot", c)
    return newdf

dfhot = oneHotEncodeColumns(dfnumeric, ["Take-out","GoodFor_lunch", "GoodFor_dinner", "GoodFor_breakfast"])
```

### Create the K-Means Model
You will use the feaures in the customer data to create a Kn-Means model with a k value of 5. This will be used to generate 5 clusters.


```python
assembler = VectorAssembler(inputCols = list(set(dfhot.columns) | set(['stars','review_count'])), outputCol="features")
train = assembler.transform(dfhot)

knum = 5
kmeans = KMeans(featuresCol=assembler.getOutputCol(), predictionCol="cluster", k=knum, seed=0)
model = kmeans.fit(train)
print "Model Created!"
```

### Get the Cluster Centers
The cluster centers are indicated as vector coordinates.


```python
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)
```

### Predict Clusters
Now that you have trained the model, you can use it to segemnt the customer data into 5 clusters and show each customer with their allocated cluster.


```python
# data set does not need to be divided to train and test
prediction = model.transform(train)
prediction.groupBy("cluster").count().orderBy("cluster").show()
```


```python
# Look at the features of each cluster

# define dictionary
customerCluster = {}
for i in range(0,knum):
    tmp = prediction.select("stars","review_count","Take-out","GoodFor_lunch", "GoodFor_dinner", "GoodFor_breakfast")\
                                    .where("cluster =" +  str(i))
    customerCluster[str(i)]= tmp
    print "Cluster"+str(i)
    customerCluster[str(i)].show(50)
```


```python

```
