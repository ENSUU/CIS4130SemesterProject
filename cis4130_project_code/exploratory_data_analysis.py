# Importing necessary libraries 
import boto3
import pandas as pd

# Getting the files inside the S3 Bucket 
s3_resource = boto3.resource('s3')
s3_bucket = s3_resource.Bucket("project-data-planes")
files = s3_bucket.objects.all()

# Getting the file names from the S3 Bucket
for file in files:
	print(file.key)

# Creating a Pandas DataFrame with all the csv files combined:
s3_location = 's3://project-data-planes/'

df_append = pd.DataFrame()

for file in files:

	if file.key == "airport.csv":
		break

df_temp = pd.read_csv(f'{s3_location}' + file.key, encoding='latin-1')
	df_append = df_append.append(df_temp, ignore_index = True)

# Preliminary descriptive statistics on dataset
for col in df.columns:
	print(f"{col}'s .describe:")
	df[col].describe()

# Checking number of null/NA values in the dataset
df.isna().sum().sum()
