import dataGenerator
import anomalyDetector

# Generate data
windowSize = 100
windowCount = 10
anomalies = [5, 7]
generator = dataGenerator.DataGenerator(windowCount, windowSize, anomalies)
data = generator.generate_data(False)

# Generate features
# detector = anomalyDetector.AnomalyDetector(anomaly)
# detector.reduceArma(data)
# detector.printFeatures()
# detector.visualizeFeatures()