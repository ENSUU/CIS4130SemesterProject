bucket = 'project-data-planes/'
filename = 'combined_airplane_dataset.csv' # Created by combining all CSV files. 
filepath = 's3a://' + bucket + filename

# Creating a Spark DataFrame from file contained in project-data-planes S3 bucket
sdf = spark.read.csv(filepath, header=True, inferSchema=True)

# Changing datatype of certain columns to float
sdf = sdf.withColumn("Distance", sdf['Distance'].cast('float'))
sdf = sdf.withColumn("TaxiIn", sdf['TaxiIn'].cast('float'))
sdf = sdf.withColumn("TaxiOut", sdf['TaxiOut'].cast('float'))
sdf = sdf.withColumn("AirTime", sdf['AirTime'].cast('float'))
sdf = sdf.withColumn("ActualElapsedTime", sdf['ActualElapsedTime'].cast('float'))
sdf = sdf.withColumn("CRSElapsedTime", sdf['CRSElapsedTime'].cast('float'))
sdf = sdf.withColumn("Delayed", sdf['Delayed'].cast('float'))

# Filtering out rows in dataset 
sdf = sdf.filter(sdf.DepTime < 2400)
sdf = sdf.filter(sdf.ArrTime < 2400)
sdf = sdf.filter(sdf.ActualElapsedTime > 0)
sdf = sdf.filter(sdf.CRSElapsedTime > 0)
sdf = sdf.filter(sdf.AirTime > 0)

# Dropping row where ActualElapsedTime is Null
sdf = sdf.filter(sdf.ActualElapsedTime.isNotNull())

# Dropping row where ArrDelay is Null
sdf = sdf.filter(sdf.ArrDelay.isNotNull())

# Dropping rows where Distance is Null
sdf = sdf.filter(sdf.Distance.isNotNull())

# Dropping rows where AirTime is Null
sdf = sdf.filter(sdf.AirTime.isNotNull())

# Dropping the TailNum column
from pyspark.sql.functions import col
sdf = sdf.drop(col("TailNum"))

# Dropping the CancellationCode column 
sdf = sdf.drop(col("CancellationCode"))

# Replacing the null values in the "____ Delay" columns with 0
sdf = sdf.na.fill(value=0)

# Saving the cleaned Spark DataFrame to the project-data-planes S3 bucket
output_file_path = "s3://project-data-planes/cleaned_combined_airplane_datset.csv"
sdf.write.options(header='True', delimiter=',').csv(output_file_path)
