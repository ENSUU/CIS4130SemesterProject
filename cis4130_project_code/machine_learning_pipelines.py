# Importing necessary libraries
from pyspark.sql.functions import *
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Bucketizer
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import *

# Linear Regression Model – Predicting Actual Flight Times

# Linear Regression Model – Using one 80-20 Random Split
# Creating an indexer for the categorical fields to be used as variables in our linear regression model (UniqueCarrier, Origin, Dest) 
indexer_linear = StringIndexer(inputCols = ["UniqueCarrier", "Origin", "Dest"], outputCols=["CarrierIndex", "OriginIndex", "DestIndex"])

# Creating a one-hot encoder for the previously indexed fields
onehot_linear = OneHotEncoder(inputCols=["CarrierIndex", "OriginIndex", "DestIndex"], outputCols=['carrier_dummy', 'origin_dummy', 'dest_dummy'])

coeff_variables_linear = ['carrier_dummy', 'origin_dummy', 'dest_dummy', 'AirTime', 'ArrDelay', 'TaxiIn', 'TaxiOut', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay'] 

# Assembling the one-hot encoded fields and other variables to be used in our model 
assembler_linear = VectorAssembler(inputCols=coeff_variables_linear, outputCol='features')

# Constructing a pipeline 
pipeline_linear = Pipeline(stages=[indexer_linear, onehot_linear, assembler_linear])

sdf_transformed = pipeline_linear.fit(sdf).transform(sdf) 

# Splitting the dataset into training and testing sets on an 80-20 split 
train_data, test_data = sdf_transformed.randomSplit([0.8, 0.2]) 

# Instantiating a LinearRegression object 
linear_regression = LinearRegression(labelCol="ActualElapsedTime")

model = linear_regression.fit(train_data) 
res = model.evaluate(test_data) 

print("MSE: ", res.meanSquaredError)
print("RMSE: ", res.rootMeanSquaredError)
print("R-Squared", res.r2)

# Saving the model to an S3 bucket
model_path = 's3://project-data-planes/flights_linear_regression_model'
model.write().overwrite().save(model_path)

# Linear Regression Model – Using Cross Validation & Obtaining Best Model 

train_data, test_data = sdf_transformed.randomSplit([0.8, 0.2]) 

linear_regression = LinearRegression(labelCol="ActualElapsedTime")

evaluator = RegressionEvaluator(labelCol="ActualElapsedTime") 

params = ParamGridBuilder()
params = params.addGrid(linear_regression.regParam, [0.01, 0.1, 1.0, 10.0]).addGrid(linear_regression.elasticNetParam, [0.0, 0.5, 1.0])
params = params.build() 

cv = CrossValidator(estimator=linear_regression, estimatorParamMaps=params, evaluator=evaluator, numFolds=3)
cv = cv.fit(train_data) 

best_model = cv.bestModel
predictions = best_model.transform(test_data) 
res_linear = best_model.evaluate(test_data) 

print("MSE: ", res_linear.meanSquaredError)
print("RMSE: ", res_linear.rootMeanSquaredError)
print("R-Squared: ", res_linear.r2)

# Saving the model to an S3 bucket
model_path = 's3://project-data-planes/flights_linear_regression_best_model'
best_model.write().overwrite().save(model_path)
	
# Logistic Regression – Predicting Flight Delays

# Creating a new column "Delayed" with either 1 (True) or 0 (False) 
sdf = sdf.withColumn("Delayed", when((sdf.ArrDelay > 0), lit(1)).otherwise(lit(0)))

# Creating an indexer for the columns with string data. 
indexer_logistic = StringIndexer(inputCols = ["UniqueCarrier", "Origin", "Dest"], outputCols=["CarrierIndex", "OriginIndex", "DestIndex"])

# Create an encoder for the string + columns that will be used for the logistic regression parameters. 
coeff_variables_logistic = ["CarrierIndex", "OriginIndex", "DestIndex", "Distance", "TaxiIn", "TaxiOut", "AirTime", "ActualElapsedTime", "CRSElapsedTime", "Year", "Month", "DayofMonth", "DayOfWeek"]
encoder_logistic = OneHotEncoder(inputCols = coeff_variables_logistic, outputCols = ["CarrierVector", "OriginVector", "DestVector", "DistanceVector", "TaxiInVector", "TaxiOutVector", "AirTimeVector", "ActualElapsedTimeVector", "CRSElapsedTimeVector", "YearVector", "MonthVector", "DayofMonthVector", "DayOfWeekVector"], dropLast=False)

# Create an assembler for the feature vectors and the columns to be used in the logistic regression model
assembler_logistic = VectorAssembler(inputCols=["CarrierVector", "OriginVector", "DestVector", "DistanceVector", "TaxiInVector", "TaxiOutVector", "AirTimeVector", "ActualElapsedTimeVector", "CRSElapsedTimeVector", "YearVector", "MonthVector", "DayofMonthVector", "DayOfWeekVector"], outputCol = "features")

# Create the pipeline
flights_pipe_logistic = Pipeline(stages=[indexer, encoder, assembler])

# Calling .fit to transform the data
transformed_sdf = flights_pipe_logistic.fit(sdf).transform(sdf)

# Splitting the data into training and test sets on a 70-30 split 
train_data, test_data = transformed_sdf.randomSplit([0.7, 0.3])

lr = LogisticRegression(labelCol="Delayed")

# Fitting/Transforming the logistic regression model on the training and testing data
lrm = lr.fit(train_data)
test_results = lrm.transform(test_data)

# Using a confusion matrix to validate the model
test_results.groupby('Delayed').pivot('prediction').count().show()

# Saving the model to an S3 bucket
model_path = 's3://project-data-planes/flights_lrm'
lrm.write().overwrite().save(model_path)
