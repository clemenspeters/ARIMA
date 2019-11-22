import datetime
import pandas as pd
import os
import terminalColors as tc

'''
Read large CSV file and sum passenger count into 30 minute buckets.
Please download raw data from: https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
'''

class TaxiDataFileProcessor:

    def __init__(
        self,
        source_file,
        dataset_name,
        pickup_index,
        passenger_index,
        write_bucket_files = False
    ):
        self.header = None
        self.time_buckets = {}
        self.source_file = source_file
        self.dataset_name = dataset_name
        self.pickup_index = pickup_index
        self.passenger_index = passenger_index
        self.write_bucket_files = write_bucket_files
        self.dir = 'taxi_data'
        self.reports_dir = '{}/reports'.format(self.dir)
        self.buckets_dir = '{}/buckets'.format(self.dir)
        # Create directories for output files
        self.create_directories(self.reports_dir)
        if write_bucket_files:
            self.create_directories(self.buckets_dir)

    def create_directories(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_bucket_name(self, pickup_date_time):
        date, *time = pickup_date_time.split(' ') # Split date and time
        hour, minute, *second = time[0].split(':') # Split hours and minutes
        # Create buckets of half an hour
        minute_bucket = '00'
        if minute >= '30':
            minute_bucket = '30'
        bucket_name = "{}_{}-{}".format(date, hour, minute_bucket)
        return bucket_name

    def new_bucket_file(self, bucket_name, passenger_count):
        if bucket_name in self.time_buckets:
            return False
        self.time_buckets[bucket_name] = int(passenger_count)
        return True

    def append_to_bucket_file(self, line, bucket_name, passenger_count):
        with open('{}/{}.csv'.format(self.buckets_dir, bucket_name), 'a') as fd:
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
        df = pd.DataFrame(list(report.items()), columns=['date_time_bucket', 'passenger_count'])
        df.sort_values(by=['date_time_bucket'], inplace=True)
        report_file = '{}/{}.csv'.format(self.reports_dir, self.dataset_name)
        df.to_csv(report_file, index=False)
        print('📂 🎉 Created report file at: {}{}{}'.format(
            tc.GREEN, report_file, tc.END
        ))

    def is_invalid_line(self, line_number, line, columns):
        if len(columns) < self.passenger_index:
            print(tc.RED)
            print("Unexpected line #{}: '{}' in {}".format(
                line_number, line, self.source_file
            ))
            print(tc.END)

            return True

        return False

    def process_file(self):
        with open(self.source_file) as f:

            print('{}Started processing {}{} at {} ⚙️ ⚙️ ⚙️'.format(
                tc.YELLOW,
                self.dataset_name, 
                tc.END,
                datetime.datetime.now()
            ))

            for i, line in enumerate(f):

                if i == 0: 
                    self.header = line
                    continue  # Skip header

                columns = line.split(',')
                if self.is_invalid_line(i + 1, line, columns):
                    continue
                pickup_date_time = columns[self.pickup_index] # Choose pickup column
                passenger_count = columns[self.passenger_index]
                bucket_name = self.get_bucket_name(pickup_date_time)

                if self.write_bucket_files:
                    self.append_to_bucket_file(
                        line, 
                        bucket_name, 
                        passenger_count
                    )
                else:
                    self.add_to_bucket(passenger_count, bucket_name)
                    
            self.write_report(self.time_buckets)

            print('{}Finshed processing {}{} at {} 🏆'.format(
                tc.GREEN,
                self.dataset_name,
                tc.END,
                datetime.datetime.now()
            ))

