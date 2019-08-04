import dataGenerator
import anomalyDetector

# Generate data
generator = dataGenerator.DataGenerator()
data = generator.getData()

# Generate features
detector = anomalyDetector.AnomalyDetector()
detector.reduceArma(data)
detector.printFeatures()