import datetime
import pandas as pd

header = 'undeined_header'
file_list = {}
write_bucket_files = False
report_file_name = 'z_buckets/report'

def get_bucket_name(pickup_date_time):
    date, *time = pickup_date_time.split(' ') # Split date and time
    hour, minute, *second = time[0].split(':') # Split hours and minutes
    # Create buckets of half an hour
    minute_bucket = '00'
    if minute >= '30':
        minute_bucket = '30'
    bucket_name = "{}_{}-{}".format(date, hour, minute_bucket)
    return bucket_name

def new_bucket_file(bucket_name):
    if bucket_name in file_list:
        return False
    file_list[bucket_name] = 0
    return True

def append_to_bucket_file(line, bucket_name):
    with open('z_buckets/{}.csv'.format(bucket_name), 'a') as fd:
        if new_bucket_file(bucket_name):
            fd.write(header)
        fd.write(line)
        file_list[bucket_name] += 1

def add_to_bucket(bucket_name):
    if bucket_name in file_list:
        file_list[bucket_name] += 1
        return
    file_list[bucket_name] = 0

def write_report(report):
    df = pd.DataFrame(list(report.items()), columns=['date_time_bucket', 'passenger_count'])
    df.sort_values(by=['date_time_bucket'], inplace=True)
    df.to_csv(report_file_name + ".csv", index=False)

def process_file(source_file):
    with open(source_file) as f:
        print("Start", datetime.datetime.now())
        for i, line in enumerate(f):
            if i == 0: 
                header = line
                continue  # Skip header
            pickup_date_time = line.split(',')[1] # Choose pickup column
            bucket_name = get_bucket_name(pickup_date_time)
            if write_bucket_files:
                append_to_bucket_file(line, bucket_name) # append to file
            else:
                add_to_bucket(bucket_name)
        write_report(file_list)
        print("End", datetime.datetime.now())

process_file("../../Downloads/yellow_tripdata_2015-01.csv")
