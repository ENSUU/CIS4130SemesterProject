# Installing packages
pip3 install pandas
pip3 install boto3
pip3 install s3fs
pip3 install aiobotocore
pip3 install matplotlib
pip3 install seaborn 

# Importing necessary libraries
import io
import pandas as pd
import s3fs
import boto3
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.stat import Correlation
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import numpy as np

# Visualization 1: Bar Chart – Number of Flights by Year

# Creating a Pandas DataFrame grouped by Year and a count
sample_df = sdf.sample(False, 0.5, 42)
pandas_df = sample_df.groupBy("Year").count().sort("Year").toPandas()

# Creating the bar chart showing the total number of flights for each year
plt.bar(pandas_df['Year'], pandas_df['count'])
plt.title('Number of Flights by Year')
plt.xlabel('Year')
plt.ylabel("Total Number of Flights (in millions)")

# Saving the plot into S3 bucket
img_data = io.BytesIO()
plt.savefig(img_data, format='png')
img_data.seek(0)

s3 = s3fs.S3FileSystem(anon=False)
with s3.open('s3://project-data-planes/year_Count2.png', 'wb') as f:
    f.write(img_data.getbuffer())

plt.clf() # To clear the plot for a new visualization

# Visualization 2: Line Graph – Total Number of Flights by Month

# Creating a Pandas DataFrame grouped by Month and respective count
sample_df2 = sdf.sample(False, 0.75, 42)
pandas_df2 = sample_df2.groupBy('Month').count().sort("Month").toPandas()

line = sns.lineplot(x="Month", y="count", data = pandas_df2)
fig = line.get_figure()

fig.savefig(img_data, format='png', bbox_inches='tight')
s3 = s3fs.S3FileSystem(anon=False)
with s3.open('s3://project-data-planes/month_Count.png', 'wb') as f:
    f.write(img_data.getbuffer())

plt.clf() # To clear the plot for a new visualization

# Visualization 3: Correlation Matrix

vector_column = "correlation_features"
numeric_columns = ['DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime', 'FlightNum', 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime', 'ArrDelay', 'DepDelay', 'Distance', 'TaxiIn', 'TaxiOut', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']

assembler = VectorAssembler(inputCols=numeric_columns, outputCol = vector_column)
sdf_vector = assembler.transform(sdf).select(vector_column) 

matrix = Correlation.corr(sdf_vector, vector_column).collect()[0][0]
correlation_matrix = matrix.toArray().tolist()

correlation_matrix_df = pd.DataFrame(data=correlation_matrix, columns=numeric_columns, index=numeric_columns)

plt.figure(figsize=(35, 10))
sns.heatmap(correlation_matrix_df, xticklabels=correlation_matrix_df.columns.values, yticklabels=correlation_matrix_df.columns.values, cmap="crest", annot=True)

# img_data = io.BytesIO()
plt.savefig(img_data, format='png', bbox_inches='tight')
img_data.seek(0)

s3 = s3fs.S3FileSystem(anon=False)
with s3.open('s3://project-data-planes/correlation_Matrix.png', 'wb') as f:
    f.write(img_data.getbuffer())

plt.clf() # To clear the plot for a new visualization

# Visualization 4: ROC Curve of the Logistic Regression Model

columns = ["UniqueCarrier", "Origin", "Dest", "Distance", "TaxiIn", "TaxiOut", "AirTime", "ActualElapsedTime", "CRSElapsedTime", "Year", "Month", "DayofMonth", "DayOfWeek", "Delayed"]
sdf_subset = sdf.select(*columns).sample(False, 0.25, 43)

# Creating an indexer for the columns with string data. 
indexer = StringIndexer(inputCols = ["UniqueCarrier", "Origin", "Dest"], outputCols=["CarrierIndex", "OriginIndex", "DestIndex"], handleInvalid="keep")

# Create an encoder for the string + columns that will be used for the logistic regression parameters. 
coeff_variables_logistic = ["CarrierIndex", "OriginIndex", "DestIndex", "Year", "Month", "DayofMonth", "DayOfWeek"]
encoder = OneHotEncoder(inputCols = coeff_variables_logistic, outputCols = ["CarrierVector", "OriginVector", "DestVector", "YearVector", "MonthVector", "DayofMonthVector", "DayOfWeekVector"], dropLast=True, handleInvalid="keep")

# Create an assembler for the feature vectors and the columns to be used in the logistic regression model
assembler = VectorAssembler(inputCols=["CarrierVector", "OriginVector", "DestVector", "Distance", "TaxiIn", "TaxiOut", "AirTime", "ActualElapsedTime", "CRSElapsedTime", "YearVector", "MonthVector", "DayofMonthVector", "DayOfWeekVector"], outputCol = "features")

flights_pipe = Pipeline(stages = [indexer, encoder, assembler])

sdf_subset_transformed = flights_pipe.fit(sdf_subset).transform(sdf_subset) 

# Splitting data into test and training
train_data, test_data = sdf_subset_transformed.randomSplit([0.7,0.3])

# Creating the LogisticRegression Estimator + Logistic Regression Model
lr = LogisticRegression(labelCol="Delayed")
lrm = lr.fit(train_data) 

# Creating ROC Curve Visualization
plt.figure(figsize=(10,10))
plt.plot([0,1], [0,1], 'r--')
plt.plot(lrm.summary.roc.select('FPR').collect(), lrm.summary.roc.select('TPR').collect())
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve")

# Evaluating the logistic regression model using AUC 
evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC", labelCol="Delayed")
predictions = lrm.transform(test_data) 
auc = evaluator.evaluate(predictions) 
print(auc) 

# Saving visualization to S3 Bucket
img_data = io.BytesIO()
plt.savefig(img_data, format='png', bbox_inches='tight')
img_data.seek(0)
s3 = s3fs.S3FileSystem(anon=False)
with s3.open('s3://project-data-planes/ROC_Curve.png', 'wb') as f:
    f.write(img_data.getbuffer())
