import dataGenerator
import anomalyDetector

def get_anomaly(windowCount, windowSize):
    anomalyStart = int(windowCount / 2) * windowSize
    anomalyEnd = anomalyStart + windowSize
    return [anomalyStart, anomalyEnd]
    # return [100, 150]

# Generate data
windowSize = 100
windowCount = 1000
anomaly = get_anomaly(windowCount, windowSize)
generator = dataGenerator.DataGenerator(windowCount, windowSize, anomaly)
data = generator.generate_data()
generator.visualize()

# Generate features
# detector = anomalyDetector.AnomalyDetector(anomaly)
# detector.reduceArma(data)
# detector.printFeatures()
# detector.visualizeFeatures()