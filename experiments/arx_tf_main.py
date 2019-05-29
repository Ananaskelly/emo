import numpy as np
import tensorflow as tf


from core.arx_dataset import ARXDataset
from core.arx_tf_model import ARXTFModel
from core.arx_batcher import ARXBatcher
from core.arx_predictor import ARXPredictor
from core.metrics import CCC_metric


if __name__ == '__main__':
    ds = ARXDataset('../data/ARX/REC_features_egemaps_10Hz.csv', '../data/ARX/REC_labels_arousal_shifted.csv',
                    '../data/ARX/REC_labels_valence_shifted.csv')

    x_train_by_file, y_train_by_file = ds.train_set_by_file
    x_test_by_file, y_test_by_file = ds.test_set_by_file
    # 70
    order = 3
    context = 5

    batcher = ARXBatcher(x_train_by_file, y_train_by_file, context=context, order=order)
    print('Data prepared..')

    model = ARXTFModel(context=context, order=order)
    model.build_model()

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    predictor = ARXPredictor(model=model, session=sess, context=context, order=order)

    batch_size = 28
    num_train_samples = batcher.num_train_samples

    epoch_steps = num_train_samples // batch_size
    num_epoch = 15

    for epoch in range(num_epoch):
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
        print('Mean epoch loss: {}'.format(mean_epoch_loss/epoch_steps))
        batcher.shuffle()

        mean_CCC = 0
        num_test_files = len(x_test_by_file)
        for idx, file in enumerate(x_test_by_file):
            y_pred = predictor.file_predict(file, y_test_by_file[idx])

            cur_CCC = CCC_metric(y_pred, y_test_by_file[idx][context - 1:, 0])
            mean_CCC += cur_CCC

        print('Mean CCC: {}'.format(mean_CCC / num_test_files))

    mean_CCC = 0
    num_test_files = len(x_test_by_file)
    for idx, file in enumerate(x_test_by_file):
        y_pred = predictor.file_predict(file, y_test_by_file[idx])

        cur_CCC = CCC_metric(y_pred, y_test_by_file[idx][6:, 0])
        print(cur_CCC)
        mean_CCC += cur_CCC

    print('Mean CCC: {}'.format(mean_CCC / num_test_files))
