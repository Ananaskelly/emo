import csv
import numpy as np


class ARXDataset(dict):

    def __init__(self, feats_path, ar_labels_path, val_labels_path):
        """
        Prepare feats and split them on train and test sets.

        :param feats_path:      path to feats in *.csv format
        :param ar_labels_path:  path to ar labels
        :param val_labels_path: path to val labels
        """
        super(ARXDataset, self).__init__()

        self.feats_path = feats_path
        self.ar_labels_path = ar_labels_path
        self.val_labels_path = val_labels_path

        self['train_data'] = None
        self['train_labels'] = None

        self['test_data'] = None
        self['test_labels'] = None

        self['train_data_by_file'] = None
        self['train_labels_by_file'] = None

        self['test_data_by_file'] = None
        self['test_labels_by_file'] = None

        self.feats_by_file = []
        self.labels_by_file = []

        self.read_data()

        self.std_mean_normalize()

        self.feats_by_file = np.array(self.feats_by_file)
        self.labels_by_file = np.array(self.labels_by_file)

        self.shuffle()
        self.split()

    @property
    def train_set(self):
        return self['train_data'], self['train_labels']

    @property
    def test_set(self):
        return self['test_data'], self['test_labels']

    @property
    def train_set_by_file(self):
        return self['train_data_by_file'], self['train_labels_by_file']

    @property
    def test_set_by_file(self):
        return self['test_data_by_file'], self['test_labels_by_file']

    def read_data(self):

        with open(self.feats_path, 'r') as feats_file, open(self.ar_labels_path, 'r') as ar_file, open(
                self.val_labels_path) as val_file:
            feats_reader = csv.reader(feats_file, delimiter=',')
            ar_reader = csv.reader(ar_file, delimiter=',')
            val_reader = csv.reader(val_file, delimiter=',')

            # skip header
            next(feats_reader)
            next(ar_reader)
            next(val_reader)

            curr_name = ''
            curr_feats = []
            curr_labels = []
            for row in feats_reader:
                full_name = row[0]
                name = full_name.split('_')[0]

                if name != curr_name:
                    # next(ar_reader)
                    # next(val_reader)
                    curr_name = name
                    if len(curr_feats) != 0:
                        self.feats_by_file.append(np.array(curr_feats))
                        self.labels_by_file.append(np.array(curr_labels))
                        curr_feats = []
                        curr_labels = []

                ar_row = next(ar_reader)
                val_row = next(val_reader)

                if ar_row[0] != full_name or val_row[0] != full_name:
                    # print('Mismatch! feat_name: {}, ar_name: {}, val_name: {}'
                    # .format(full_name, ar_row[0], val_row[0]))
                    while ar_row[0] != full_name:
                        ar_row = next(ar_reader)
                    while val_row[0] != full_name:
                        val_row = next(val_reader)

                feats = np.array(row[1:]).astype(np.float)
                curr_feats.append(feats)
                curr_labels.append(np.array([ar_row[1], val_row[1]]).astype(np.float))

    def std_mean_normalize(self):
        data = self.feats_by_file.copy()
        data = np.concatenate(data)

        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        data -= mean
        data /= std

        curr_id = 0
        for idx, file_feats in enumerate(self.feats_by_file):
            num_ex = file_feats.shape[0]
            self.feats_by_file[idx] = data[curr_id: curr_id + num_ex, :]
            curr_id += num_ex

    def shuffle(self):
        num_ex = len(self.feats_by_file)
        idx = np.arange(num_ex)
        np.random.shuffle(idx)

        self.feats_by_file = self.feats_by_file[idx]
        self.labels_by_file = self.labels_by_file[idx]

    def split(self, tp=0.3):
        num_ex = self.feats_by_file.shape[0]
        num_test_ex = int(num_ex*tp)

        self['train_data'] = self.feats_by_file[:num_ex-num_test_ex]
        self['train_labels'] = self.labels_by_file[:num_ex-num_test_ex]

        self['train_data_by_file'] = self.feats_by_file[:num_ex-num_test_ex]
        self['train_labels_by_file'] = self.labels_by_file[:num_ex-num_test_ex]

        self['test_data'] = self.feats_by_file[-num_test_ex:]
        self['test_labels'] = self.labels_by_file[-num_test_ex:]

        self['test_data_by_file'] = self.feats_by_file[-num_test_ex:]
        self['test_labels_by_file'] = self.labels_by_file[-num_test_ex:]

        for tup in zip([self['train_data'], self['train_labels'], self['test_data'], self['test_labels']],
                       ['train_data', 'train_labels', 'test_data', 'test_labels']):
            buffer = tup[0].tolist()
            self[tup[1]] = np.concatenate(buffer)
