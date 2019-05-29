import numpy as np


class ARXPredictor:
    def __init__(self, model, session, context=7, order=2):
        self.model = model
        self.sess = session

        self.context = context
        self.order = order

    def file_predict(self, data):
 
        predicted_array = np.zeros(shape=(len(data) - self.context + 1))

        for i in range(len(data) - self.context + 1):

            data_sample = np.array(data[i: i + self.context])
            if self.order > 0:
                if i == 0:
                    y_in = np.zeros(shape=self.order)
                elif i < self.order:
                    y_in = np.concatenate((np.zeros(shape=self.order - i), predicted_array[:i]))
                else:
                    y_in = predicted_array[i - self.order: i]
                frame_pred = self.sess.run([self.model.y_hat], {
                    self.model.x: np.expand_dims(data_sample, 0),
                    self.model.y_past: np.expand_dims(y_in, 0)
                })
            else:
                frame_pred = self.sess.run([self.model.y_hat], {
                    self.model.x: np.expand_dims(data_sample, 0)
                })
            predicted_array[i] = frame_pred[0][0]
        return predicted_array
