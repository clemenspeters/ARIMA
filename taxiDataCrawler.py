import urllib.request
import taxiDataFileProcessor

base_url = 'https://s3.amazonaws.com/nyc-tlc/trip+data/'

years = range(2009, 2020)
months = range(1,13)


def dry_run_download(url):
    try:
        with urllib.request.urlopen(url) as response:
            if(response.code == 200):
                print(response.code, url, "file exists.")
    except urllib.error.HTTPError as err:
        print(err.code, url)


def download_url(url, filename):
    try:
        with urllib.request.urlopen(url) as response:
            if(response.code == 200):
                print(response.code, "downloading", url)
                urllib.request.urlretrieve (url, filename)
                print("Completed", url)
    except urllib.error.HTTPError as err:
        print(err.code, url)


def check_files():
    for year in years:
        for month in months:
            if year == 2019 and month > 6: # 2019-06 is the latest dataset.
                return
            formatted_month = "{:02d}".format(month) # Add leading zeroes
            dataset_name = "yellow_tripdata_{}-{}".format(year, formatted_month)
            filename = "{}.csv".format(dataset_name)
            url = "{}{}".format(base_url, filename)
            # dry_run_download(url)
            download_url(url, filename)
            process_data_file(filename, dataset_name)


def process_data_file(data_source_file, dataset_name):

    yellowTaxiProcessor = taxiDataFileProcessor.TaxiDataFileProcessor(
        source_file = data_source_file,
        dataset_name = dataset_name,
        pickup_index = 1,
        passenger_index = 3
    )

    yellowTaxiProcessor.process_file()


check_files()


