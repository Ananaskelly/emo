import numpy as np


class RFDataset:

    def __init__(self, data_by_file, labels_by_file, context=7):
        self.init_data = data_by_file
        self.init_labels = labels_by_file

        self.context = context

        self.data = []
        self.output_labels = []

        self.prepare()

    @property
    def train_set(self):
        return self.data, self.output_labels

    def prepare(self):

        for idx, file_feats in enumerate(self.init_data):
            num_feats, feats_dim = file_feats.shape

            for i in range(0, num_feats - self.context):
                feats = file_feats[i: i + self.context]
                y_past = np.concatenate((self.init_labels[idx][i: i + self.context - 1, 0], [0]))
                feats = np.hstack((feats, np.expand_dims(y_past, 1)))
                self.data.append(feats)

                self.output_labels.append(self.init_labels[idx][i + self.context - 1, 0])

        self.data = np.stack(self.data)
        
        self.output_labels = np.stack(self.output_labels)
