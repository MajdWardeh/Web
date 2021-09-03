import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss, MeanAbsoluteError
from tensorflow.keras import backend as k
from scipy.special import binom

class BezierLoss(Loss):

    def __init__(self, numOfControlPoints=5, dimentions=3, gamma=0.5):
        # n is the order of the Bezier.
        super().__init__()
        self.gamma = gamma
        self.numOfControlPoints = numOfControlPoints
        self.dimentions = dimentions
        self.mae = MeanAbsoluteError()

    def call(self, y_true, y_pred):

        cp_true = tf.reshape(y_true, (-1, self.numOfControlPoints, self.dimentions))
        cp_hat = tf.reshape(y_pred, (-1, self.numOfControlPoints, self.dimentions))

        y_true_diff = cp_true[:, 1:, :] - cp_true[:, :-1, :]
        y_pred_diff = cp_hat[:, 1:, :] - cp_hat[:, :-1, :]
        
        MAE_cp = self.mae(y_true, y_pred)
        MAE_cp_diff = self.mae(y_true_diff, y_pred_diff)

        # bs = tf.shape(y_true)[0]
        # cp_true = tf.reshape(y_true, (bs, 4, 3))
        # # cp_true = tf.concat([tf.zeros(shape=(bs, 1, 3)), cp_true], axis=1)
        # cp_pred = tf.reshape(y_pred, (bs, 4, 3))
        # cp_pred = tf.concat([tf.zeros(shape=(bs, 1, 3)), cp_pred], axis=1)
        # P_true = tf.matmul(self.A, cp_true)
        # P_pred = tf.matmul(self.A, cp_pred)
        # MAE_curve = self.mae(P_true, P_pred)
        # MAE_cp = self.mae(y_true, y_pred)

        # print(MAE_cp_diff)
        # print(MAE_cp)

        return self.gamma * MAE_cp + (1-self.gamma) * MAE_cp_diff

        # abs_error = tf.math.abs(P_true-P_pred)
        # MAExyz = tf.reduce_mean(abs_error, 0)
        # MAE = tf.reduce_mean(MAExyz)
        # return MAE


def main():
    numOfCP = 5
    dimentions = 3
    batchSize = 20
    cp1 = np.random.rand(batchSize, numOfCP, dimentions)
    cp2 = np.random.rand(batchSize, numOfCP, dimentions)
    print(cp1)
    print(cp2)

    cp1 = tf.convert_to_tensor(cp1, dtype=tf.float64)
    cp2 = tf.convert_to_tensor(cp2, dtype=tf.float64)

    diffCP1 = cp1[:, 1:, :] - cp1[:, :-1, :]
    diffCP2 = cp2[:, 1:, :] - cp2[:, :-1, :]
    diffCP1 = tf.convert_to_tensor(diffCP1, dtype=tf.float64)
    diffCP2 = tf.convert_to_tensor(diffCP2, dtype=tf.float64)

    maeLoss = MeanAbsoluteError()

    diffMAE = maeLoss(diffCP1, diffCP2)
    print(diffMAE)

    mae = maeLoss(cp1, cp2)
    print(mae)

    bezierLoss = BezierLoss(numOfCP, dimentions)
    bezierLoss.call(tf.reshape(cp1, [-1]), tf.reshape(cp2, [-1]))




if __name__ == '__main__':
    main()

