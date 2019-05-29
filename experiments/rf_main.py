import numpy as np
from sklearn.ensemble import RandomForestRegressor


from core.rf_dataset import RFDataset
from core.arx_dataset import ARXDataset
from core.rf_predictor import RFPredictor
from core.metrics import CCC_metric


if __name__ == '__main__':
    hardcoded_path_feats = '../data/ARX/REC_features_egemaps_10Hz.csv'
    hardcoded_path_ar = '../data/ARX/REC_labels_arousal_shifted.csv'
    hardcoded_path_val = '../data/ARX/REC_labels_valence_shifted.csv'

    ds = ARXDataset(hardcoded_path_feats, hardcoded_path_ar, hardcoded_path_val)

    x_train_by_file, y_train_by_file = ds.train_set_by_file
    x_test_by_file, y_test_by_file = ds.test_set_by_file

    context = 7
    rf_ds = RFDataset(x_train_by_file, y_train_by_file, context)

    data, labels = rf_ds.train_set
    n, context, num_feats = data.shape
    data = np.reshape(data, (n, context*num_feats))
    rf_engine = RandomForestRegressor(n_estimators=100, max_depth=10,
                                      random_state=42, verbose=1, n_jobs=5)
    rf_engine.fit(data, labels)

    rf_predictor = RFPredictor(rf_engine, context)

    mean_CCC = 0
    num_test_files = len(x_test_by_file)
    for idx, file in enumerate(x_test_by_file):

        y_pred = rf_predictor.predict(file)
        cur_CCC = CCC_metric(y_pred, y_test_by_file[idx][context - 1:, 0])
        mean_CCC += cur_CCC

    print('Mean CCC on test set: {}'.format(mean_CCC / num_test_files))
