# Processing pipeline for NYC Taxi datasets

The original datasets can be found at the NYC Taxi and Limousine Commission  
[Website](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

It is grouped by months, so there is a separate CSV file for every month.
The data looks like this:

```csv
VendorID,lpep_pickup_datetime,Lpep_dropoff_datetime,Store_and_fwd_flag,RateCodeID,Pickup_longitude,Pickup_latitude,Dropoff_longitude,Dropoff_latitude,Passenger_count,Trip_distance,Fare_amount,Extra,MTA_tax,Tip_amount,Tolls_amount,Ehail_fee,improvement_surcharge,Total_amount,Payment_type,Trip_type
2,2015-01-01 00:34:42,2015-01-01 00:38:34,N,1,-73.922592163085938,40.754528045654297,-73.91363525390625,40.765522003173828,1,.88,5,0.5,0.5,0,0,,0.3,6.3,2,1,,
2,2015-01-01 00:34:46,2015-01-01 00:47:23,N,1,-73.952751159667969,40.677711486816406,-73.981529235839844,40.658977508544922,1,3.08,12,0.5,0.5,0,0,,0.3,13.3,2,1,,
1,2015-01-01 00:34:44,2015-01-01 00:38:15,N,1,-73.843009948730469,40.71905517578125,-73.846580505371094,40.711566925048828,1,.90,5,0.5,0.5,1.8,0,,0,7.8,1,1,,
```

I found this dataset thanks to Numenta who processed it to group the  
passenger count by 30 minutes buckets in their [repo](https://github.com/numenta/NAB/blob/master/data/realKnownCause/nyc_taxi.csv), so that it looks like this:

```csv
timestamp,value
2014-07-01 00:00:00,10844
2014-07-01 00:30:00,8127
2014-07-01 01:00:00,6210
2014-07-01 01:30:00,4656
2014-07-01 02:00:00,3820
2014-07-01 02:30:00,2873
```

Since I found the processed data only for one year (2014-07-01 until 2014-07-01)  
and [raw data](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) is available for 10 years I built my own processing pipeline.

The idea is simple: process every line of the raw data and add the passenger  
count to the accoring time-bucket. Since we have more than 120 (monthly) files
with a total of over 200GB available, I came up with a processing pipeline which  
allows to process many files in parallel (in the case of the yellow taxi dataset  
more than 120 files). This allows to process the more than 200GB of data in less
than 10 Minutes on AWS Lambda and s3 with very little server cost.

There are two implementations:

## External data

The data is streamed from the original (external) s3 bucket and directly  
processed.

### Advantage

Easy to setup, little s3 storage usage in your own AWS  
account (only results are stored, no raw data).

### Disadvantage

In case the original raw data is removed, we loose access (dependency on third  
party).

## Internal data

The data is streamed from the original (external) s3 bucket and directly  
processed.

### Advantage

In case the original raw data is removed, we have it save in our own s3 bucket  
(no dependency on third party).

### Disadvantage

Slightly more complex to setup (extra download step), over 200GB s3 storage  
usage in your own AWS account (all raw data is duplicated to your s3 bucket).
