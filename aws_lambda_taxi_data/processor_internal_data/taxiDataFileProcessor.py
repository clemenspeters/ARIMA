import datetime
# import pandas as pd
# import os
from smart_open import open
import terminalColors as tc
from io import StringIO
import boto3

'''
Read large CSV file and sum passenger count into 30 minute buckets.
Please download raw data from: https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
'''

class TaxiDataFileProcessor:

    def __init__(
        self,
        s3bucket,
        key,
        pickup_index,
        passenger_index,
        write_bucket_files = False
    ):
        self.header = None
        self.time_buckets = {}
        self.s3bucket = s3bucket
        self.key = key
        self.dataset_name = key.split('/')[1].split('.')[0]
        self.pickup_index = pickup_index
        self.passenger_index = passenger_index
        self.write_bucket_files = write_bucket_files
        self.dir = 'taxi_data'
        self.reports_dir = '{}/reports'.format(self.dir)
        self.s3buckets_dir = '{}/buckets'.format(self.dir)
        # Create directories for output files
        # self.create_directories(self.reports_dir)
        # if write_bucket_files:
        #     self.create_directories(self.s3buckets_dir)

    # def create_directories(self, directory):
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)

    def get_bucket_name(self, pickup_date_time):
        date, *time = pickup_date_time.split(' ') # Split date and time
        hour, minute, *second = time[0].split(':') # Split hours and minutes
        # Create buckets of half an hour
        minute_bucket = '00'
        if minute >= '30':
            minute_bucket = '30'
        bucket_name = "{} {}:{}".format(date, hour, minute_bucket)
        return bucket_name

    def new_bucket_file(self, bucket_name, passenger_count):
        if bucket_name in self.time_buckets:
            return False
        self.time_buckets[bucket_name] = int(passenger_count)
        return True

    def append_to_bucket_file(self, line, bucket_name, passenger_count):
        with open('{}/{}.csv'.format(self.s3buckets_dir, bucket_name), 'a') as fd:
            if self.new_bucket_file(bucket_name, passenger_count):
                fd.write(self.header)
            fd.write(line)
            self.time_buckets[bucket_name] += int(passenger_count)

    def add_to_bucket(self, passenger_count, bucket_name):
        if bucket_name in self.time_buckets:
            self.time_buckets[bucket_name] += int(passenger_count)
            return
        self.time_buckets[bucket_name] = int(passenger_count)

    def write_report(self, report):
        report_file = 'report/report_{}.csv'.format(self.dataset_name)
        print("Creating report...")
        output = StringIO()
        output.write('date_time_bucket,passenger_count\n')
        for key in sorted(report.keys()):
            output.write("{},{}\n".format(key, report[key]))
        contents = output.getvalue()
        # print(contents)
        s3_resource = boto3.resource('s3')
        s3_resource.Object(
            self.s3bucket, 
            report_file
        ).put(Body=contents)

        print('📂 🎉 Created report file at: {}{}{}'.format(
            tc.GREEN, report_file, tc.END
        ))

    def is_invalid_line(self, line_number, line, columns):
        if len(columns) < self.passenger_index:
            print(tc.RED)
            print("Unexpected line #{}: '{}' in {}".format(
                line_number, line, self.s3bucket
            ))
            print(tc.END)

            return True

        return False

    def process_file(self):
        print('{}Started processing {}{} at {} ⚙️ ⚙️ ⚙️'.format(
            tc.YELLOW,
            self.dataset_name, 
            tc.END,
            datetime.datetime.now()
        ))
        lineCount = 0
        for line in open('s3://{}/{}'.format(self.s3bucket, self.key)):
            # print(lineCount)
            # print(repr(line))

            if lineCount == 0: 
                self.header = line
                lineCount += 1
                continue  # Skip header

            columns = line.split(',')
            if self.is_invalid_line(lineCount + 1, line, columns):
                lineCount += 1
                continue
            pickup_date_time = columns[self.pickup_index] # Choose pickup column
            passenger_count = columns[self.passenger_index]
            bucket_name = self.get_bucket_name(pickup_date_time)

            # if self.write_bucket_files:
            #     self.append_to_bucket_file(
            #         line, 
            #         bucket_name, 
            #         passenger_count
            #     )
            # else:
            #     self.add_to_bucket(passenger_count, bucket_name)
            self.add_to_bucket(passenger_count, bucket_name)

            lineCount += 1

            # if lineCount > 100:
            #     break
                
        self.write_report(self.time_buckets)

        print('{}Finshed processing {}{} at {} 🏆'.format(
            tc.GREEN,
            self.dataset_name,
            tc.END,
            datetime.datetime.now()
        ))

