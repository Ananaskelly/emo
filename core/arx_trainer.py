import tensorflow as tf


from core.arx_dataset import ARXDataset
from core.arx_tf_model import ARXTFModel
from core.arx_batcher import ARXBatcher
from core.arx_predictor import ARXPredictor
from core.metrics import CCC_metric


class ARXTrainer:

    def __init__(self):
        hardcoded_path_feats = '../data/ARX/REC_features_egemaps_10Hz.csv'
        hardcoded_path_ar = '../data/ARX/REC_labels_arousal_shifted.csv'
        hardcoded_path_val = '../data/ARX/REC_labels_valence_shifted.csv'

        self.ds = ARXDataset(hardcoded_path_feats, hardcoded_path_ar, hardcoded_path_val)

        self.x_train_by_file, self.y_train_by_file = self.ds.train_set_by_file
        self.x_test_by_file, self.y_test_by_file = self.ds.test_set_by_file

        self.x_valid_sample = self.x_train_by_file[-1]
        self.y_valid_sample = self.y_train_by_file[-1]

        self.x_train_by_file = self.x_train_by_file[:-1]
        self.y_train_by_file = self.y_train_by_file[:-1]

        # common params
        self.batch_size = 28
        self.max_num_epoch = 15

    def train_model(self, order, context, val_to_predict=0):
        """

        :param val_to_predict: for training predict arousal or valence value
        :return:
        """
        
        val = val_to_predict

        print('Start testing model with order: {}, context: {}'.format(order, context))

        batcher = ARXBatcher(self.x_train_by_file, self.y_train_by_file, context=context, order=order)

        model = ARXTFModel(context=context, order=order)
        model.build_model()

        init = tf.global_variables_initializer()

        sess = tf.Session()
        sess.run(init)

        predictor = ARXPredictor(model=model, session=sess, context=context, order=order)

        num_train_samples = batcher.num_train_samples
        epoch_steps = num_train_samples // self.batch_size

        best_valid_CCC = -1
        mean_train_CCC = -1
        for epoch in range(self.max_num_epoch):
            mean_epoch_loss = 0
            for step in range(epoch_steps):
                data, y_in, y_out = batcher.next_batch()
                loss, _, y_hat = sess.run([model.loss, model.opt, model.y_hat], {
                    model.x: data,
                    model.y_past: y_in,
                    model.y: y_out
                })
                # print(y_hat[0], y_out[0])
                mean_epoch_loss += loss
                # if step % 100 == 0:
                #    print('Epoch: {}, step: {}, loss: {}'.format(epoch, step, loss))
            print('Mean epoch loss: {}'.format(mean_epoch_loss / epoch_steps))
            batcher.shuffle()

            mean_CCC = self.calc_metric_valid(predictor, context, val)

            if mean_CCC > best_valid_CCC:
                best_valid_CCC = mean_CCC
                mean_train_CCC = self.calc_metric(predictor, context, val)

        print('best validation CCC: {}\ntest CCC: {}'.format(best_valid_CCC, mean_train_CCC))
        return mean_train_CCC

    def calc_metric(self, predictor, context, val):
        mean_CCC = 0
        num_test_files = len(self.x_test_by_file)
        for idx, file in enumerate(self.x_test_by_file):

            y_pred = predictor.file_predict(file)

            cur_CCC = CCC_metric(y_pred, self.y_test_by_file[idx][context - 1:, val])
            mean_CCC += cur_CCC

        mean_CCC /= num_test_files
        print('Mean CCC: {}'.format(mean_CCC))

        return mean_CCC

    def calc_metric_valid(self, predictor, context, val):

        y_pred = predictor.file_predict(self.x_valid_sample)
        cur_CCC = CCC_metric(y_pred, self.y_valid_sample[context - 1:, val])

        print('validation CCC: {}'.format(cur_CCC))

        return cur_CCC
