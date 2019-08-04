import statsmodels.api as sm

class AnomalyDetector:
    def __init__(self):
        self.windowSize = 50
        self.stride = self.windowSize / 2
        self.features = []

    def reduceArma(self, data):
        windows = self.getWindows(data)
        self.features = self.getParameters(windows, self.getArmaParams)

    def getWindows(self, data):
        result = []
        start = 0
        end = self.windowSize
        while (end <= data.size):
            result.append(data[start: end])
            start = int( start + self.stride)
            end = int( end + self.stride)
        return result

    def getParameters(self, windows, encoderFunction):
        parameters = []
        for window in windows:
            params = encoderFunction(window)
            parameters.append(params)
        return parameters

    def getArmaParams(self, window):
        model = sm.tsa.ARMA(window, (2, 2)).fit(trend='nc', disp=0)
        return model.params

    def printFeatures(self):
        for feature in self.features:
            print(feature)
