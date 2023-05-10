import sys
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from smart_open import open
import boto3
# import s3fs


def handler(event, context):
    # Example file path or URL
    file_path_or_url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-01.parquet'
    # file_path_or_url = 'yellow_tripdata_2009-01.parquet'
    # file_path_or_url = '~/Downloads/yellow_tripdata_2022-01.parquet'

    file_name = file_path_or_url.rsplit('/', 1)[-1]


    # Check if the file path is a local path or a URL
    if file_path_or_url.startswith('http'):
        url = file_path_or_url
        print('Read parquet file from url: {}'.format(url))
        # Open the Parquet file using smart_open
        with open(url, 'rb') as f:
            # Read the file into memory
            parquet_file = pq.ParquetFile(f)
        # Read the data into a pandas dataframe
        df = parquet_file.read().to_pandas()

    else:
        file_path = file_path_or_url
        print('Read parquet file from local path {}'.format(file_path))
        df = pd.read_parquet(file_path, engine='pyarrow')

    pickup_key = df.columns[1] # Can be 'Trip_Pickup_DateTime' or 'tpep_pickup_datetime'
    passenger_count_key = df.columns[3] # Can be 'Passenger_Count' or 'passenger_count'

    print('pickup_key: {}, passenger_count_key: {}'.format(pickup_key, passenger_count_key))


    # Convert the date/time column to a datetime object
    df[pickup_key] = pd.to_datetime(df[pickup_key])

    # Group the data by 30 minute intervals and sum the Passenger_Count column
    df = df.groupby(pd.Grouper(key=pickup_key, freq='30min')).agg({passenger_count_key: 'sum'})

    # Write the processed data back to a Parquet file in the S3 bucket
    # result_file_path='output/{}'.format(file_name)
    # result_file_path=file_name
    # print('Write to file: {}'.format(result_file_path))
    # df.to_parquet(result_file_path, engine='pyarrow')

    # # Write the Parquet file to an S3 bucket
    # s3 = s3fs.S3FileSystem()
    # with s3.open('s3://timeseries-anomaly/2023/{}'.format(result_file_path), 'wb') as f:
    #     with open(result_file_path, 'rb') as fp:
    #         f.write(fp.read())

    # Convert the DataFrame to a PyArrow table
    table = pa.Table.from_pandas(df)

    # Define the S3 bucket and file name for the Parquet file
    bucket = 'timeseries-anomaly'
    file_path = '2023/{}'.format(file_name)

    # Set up the S3 file system
    fs = pa.fs.S3FileSystem()

    # Open a PyArrow ParquetWriter object for the S3 file
    writer = pq.ParquetWriter('{}/{}'.format(bucket, file_path), table.schema, filesystem=fs)

    # Write the PyArrow table to the Parquet file
    writer.write_table(table)

    # Close the ParquetWriter object
    writer.close()


# handler('', '')