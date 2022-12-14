# Need to set up AWS Command Line Interface in an EC2 instance. 

# Creating a bucket in Amazon S3 to hold project data
aws s3api create-bucket --bucket project-data-planes --region us-east-2 --create-bucket-configuration LocationConstraint=us-east-2

# Installing the Kaggle Command Line Interface for Python 3
pip3 install kaggle 

# Followed the instructions posted on Blackboard to create a new API token in Kaggle, create a directory for Kaggle and edit the kaggle_api_extended.py file.

# Fetching the dataset, sending the file to standard output, piping this output to the aws s3 cp command, and directing it to the created S3 bucket 
kaggle datasets download --quiet -d wenxingdi/data-expo-2009-airline-on-time-data -p - | aws s3 cp - s3://project-data-planes/archive.zip

# Unzipping the dataset files in created S3 bucket
import zipfile 
import boto3
from io import BytesIO

bucket = “project-data-planes” 
zipfile_to_unzip = “archive.zip” 
s3_client = boto3.client(‘s3’, use_ssl=False) 
s3_resource = boto3.resource(‘s3’) 

# Create a zip object that represents the zip file 
zip_obj = s3_resource.Object(bucket_name=bucket, key=zipfile_to_unzip)
buffer = BytesIO(zip_obj.get()[“Body”].read())
z = zipfile.ZipFile(buffer) 

# Loop through all the files contained in the zip archive 
for filename in z.namelist():
    s3_resource.meta.client.upload_fileobj(z.open(filename), Bucket=bucket, Key=f’{filename}’) 
