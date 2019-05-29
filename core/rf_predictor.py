import numpy as np


class RFPredictor:
    def __init__(self, classifier, context):
        self.classifier = classifier
        self.context = context

    def predict(self, data):

        predicted_array = np.zeros(shape=(len(data) - self.context + 1))
        for i in range(len(data) - self.context + 1):

            data_sample = np.array(data[i: i + self.context])
            if i == 0:
                y_in = np.zeros(shape=self.context)
            elif i < self.context:
                y_in = np.concatenate((np.zeros(shape=self.context - i), predicted_array[:i]))
            else:
                y_in = predicted_array[i - self.context: i]

            y_in = np.expand_dims(np.concatenate((y_in, [0])), 1)
            data_sample = np.ravel(np.hstack((data_sample, y_in)))

            y_hat = self.classifier.predict(data_sample)
            predicted_array[i] = y_hat

        return predicted_array
