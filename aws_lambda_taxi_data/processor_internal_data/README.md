# AWS lambda data processor (Internal s3 bucket data)

Please read the [general desciption](../README.md) in the parent directory first.  

![Infrastructure](./img/infrastructure_serverless_data_processing_internal_data_scale.png)
  
The **publisher Lambda** function sends all filenames to SNS.  

The **downloader Lambda** is triggered by the SNS message and copies the data from  
the external s3 bucket to your own s3 bucket.  

The **processor Lambda** is triggered by the new file in your s3 bucket.
It streams the (copied) raw data from the you own s3 bucket.  
Result files (containing the 30 minute aggregations) are stored in your own  
AWS account as well (same s3 bucket different directory).

This is highly scalable as well since both the **downloader** and the  
**processor** can run in parallel.
