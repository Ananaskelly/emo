import numpy as np
from scipy.stats import pearsonr


def CCC_metric(ser1, ser2):
    """
    This function calculates concordance correlation coefficient of two series: 'ser1' and 'ser2'
    """
    if ser1.shape != ser2.shape:
        print("Series have different lengths")
        return None
    else:
        CC = pearsonr(ser1, ser2)
        CCC = 2*CC[0]*np.std(ser1)*np.std(ser2)/(np.var(ser1) + np.var(ser2) + (np.mean(ser1, axis=0)
                                                                                - np.mean(ser2, axis=0)) ** 2)
    return CCC
