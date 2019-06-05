import numpy as np


class ARXBatcher:

    def __init__(self, data_by_file, labels_by_file, order=2, context=7, batch_size=28):
        self.init_data = data_by_file
        self.init_labels = labels_by_file

        self.order = order
        self.context = context

        self.data = []
        self.input_labels = []
        self.output_labels = []
        self.current_idx = 0
        self.batch_size = batch_size

        self.prepare()

    @property
    def num_train_samples(self):
        return self.data.shape[0]

    def prepare(self, val_to_predict=0):
        """

        :param val_to_predict: for generating set for valence or for arousal prediction
        :return:
        """
        val = val_to_predict
        for idx, file_feats in enumerate(self.init_data):
            num_feats, feats_dim = file_feats.shape

            if self.order >= self.context:
                start_idx = self.order - self.context + 1
            else:
                start_idx = 0
            for i in range(start_idx, num_feats - self.context):
                self.data.append(file_feats[i:i + self.context])
                self.input_labels.append(self.init_labels[idx][i + self.context - 1 - self.order:i + self.context - 1,
                                         val])
                self.output_labels.append(self.init_labels[idx][i + self.context - 1, val])

        self.data = np.stack(self.data)
        self.input_labels = np.stack(self.input_labels)
        self.output_labels = np.stack(self.output_labels)

    def next_batch(self):
        batch_data, batch_in_y, batch_out_y = self.data[self.current_idx: self.current_idx + self.batch_size], \
               self.input_labels[self.current_idx: self.current_idx + self.batch_size], \
               self.output_labels[self.current_idx: self.current_idx + self.batch_size]

        self.current_idx += 1
        return batch_data, batch_in_y, batch_out_y

    def shuffle(self):
        idx = np.arange(self.num_train_samples)
        np.random.shuffle(idx)
        self.data = self.data[idx]
        self.input_labels = self.input_labels[idx]
        self.output_labels = self.output_labels[idx]
        self.current_idx = 0
