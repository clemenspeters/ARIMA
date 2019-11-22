import taxiDataFileProcessor

dataset_name = "yellow_tripdata_2015-01"
data_source_file = '../../Downloads/{}.csv'.format(dataset_name)

yellowTaxiProcessor = taxiDataFileProcessor.TaxiDataFileProcessor(
    source_file = data_source_file,
    dataset_name = dataset_name,
    pickup_index = 1,
    passenger_index = 3
)

yellowTaxiProcessor.process_file()
