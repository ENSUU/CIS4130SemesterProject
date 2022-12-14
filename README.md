# CIS 4130 Semester Project - *"Data Expo 2009: Airline on Time Data" Machine Learning Pipeline* 

*"Data Expo 2009: Airline on Time Data" Machine Learning Pipeline* is my submission for the semester project while taking the course CIS 4130 - Big Data Technologies. 

Submitted by: **Eunsu (David) Kim** 

## Project Description

**<ins>Objective:</ins>** To predict flight delays and the actual duration of scheduled flights. 

Using the "Data Expo 2009: Airline on Time Data" found on Kaggle.com, I built a complete machine learning pipeline which incorporates big data technologies using a cloud infrastructure. 

The project can be split into the following steps: downloading the “Data Expo 2009: Airline On Time Data” dataset from Kaggle, exploring the data (deriving descriptive statistics on the data), cleaning the data, creating the machine learning pipeline (building the machine learning models) and concludes with visualizations.

### Downloading the Dataset 
To download the data directly from Kaggle to my designated Amazon S3 bucket for this project, I had to configure the AWS Command Line Interface (CLI) on an EC2 instance, install and configure the Kaggle API and make a few edits to the Kaggle API source code to allow writing to standard output. By following these steps successfully, I was able to download the Kaggle dataset directly into my S3 bucket without downloading the CSV files directly.

### Exploratory Data Analysis
During the exploratory data analysis step, I learned about how the dataset is split into 22 CSV files, with each containing flights that occurred between the years 1987 to 2008. Using df.describe(), I was able to generate preliminary descriptive statistics on the dataset. From these descriptive statistics, I found that instead of the dataset containing flights between October 2008 to April 2008 as specified in the dataset description, the earliest record in the dataset is January 1987 and the latest record in the dataset is October 2008. Moreover, I found that some of the time variables such as ArrTime and DepTime had outliers as these values should only be between 0 to 2359. Using df.isna().sum().sum(), I discovered that there are 58,769,088 missing fields in the dataset which must be cleaned before any processing is done. 

### Cleaning the Dataset 
To clean the dataset, I had removed the rows with ArrTime and DepTime values greater than 2400, removed any rows where ActualElapsedTime, CRSElapsedTime, and AirTime values were null, and dropped the columns TailNum and CancellationCode. Additionally, I had type casted the Distance, TaxiIn, TaxiOut, AirTime, ActualElapsedTime, CRSElapsedTime, and Delayed columns to the float datatype. The remaining null values present in the dataset were present in the CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, and LateAircraftDelay columns. Under the assumption that null values for these columns signifies no delays occurred, I concluded this step by replacing the remaining null values with a placeholder value of 0. 

### Machine Learning Pipelines Creation 
With the dataset cleaned, I began by constructing a linear regression model to predict actual flight times using the features UniqueCarrier, Origin, Dest, AirTime, ArrDelay, TaxiIn, TaxiOut, CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, and LateAircraftDelay. Both the model constructed using one random 80-20 split and the best model from a 3-fold cross validation were evaluated based on the mean squared error, root mean squared error, and R-squared statistics. The second model I chose to construct is a logistic regression model to predict whether a flight will be delayed. The features for the logistic regression model are UniqueCarrier, Origin, Dest, Distance, TaxiIn, TaxiOut, AirTime, ActualElapsedTime, CRSElapsedTime, Year, Month, DayofMonth and DayOfWeek. To evaluate this model, I created a confusion matrix and calculated the precision, accuracy, and recall. 

### Visualizations
For the visualization step, I created a bar chart showing how much flights took place in each year, a line graph to show flight frequencies over the months of the year, a correlation matrix for the numeric variables in the dataset and an ROC curve for the logistic regression model.

## Challenge Encountered 
One of the challenges encountered during the project was how to deal with the null values present in the data. Although columns with too many null values (TailNum and CancellationCode) were dropped, handling null values present in CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, and LateAircraftDelay columns could be interpreted as not applicable or an error while the data was collected. Ultimately, I decided to assume the latter and replaced these null values with a placeholder value of 0. 
