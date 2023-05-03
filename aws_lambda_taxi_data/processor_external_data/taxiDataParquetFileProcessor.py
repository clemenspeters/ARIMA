import pandas as pd
from smart_open import open
import pyarrow.parquet as pq

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
    # pickup_key = 'Trip_Pickup_DateTime'
    # passenger_count_key = 'Passenger_Count'
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
print('Write to file output/{}'.format(file_name))
df.to_parquet('output/{}'.format(file_name), engine='pyarrow')
