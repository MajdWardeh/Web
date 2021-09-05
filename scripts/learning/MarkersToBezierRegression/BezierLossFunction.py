import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss, MeanAbsoluteError, MeanSquaredError, Huber
from tensorflow.keras import backend as k
from scipy.special import binom

class BezierLoss(Loss):

    def __init__(self, numOfControlPoints=5, dimentions=3, gamma=0.5, config=None):
        # n is the order of the Bezier.
        super().__init__()
        self.gamma = gamma
        self.numOfControlPoints = numOfControlPoints
        self.dimentions = dimentions
        self.loss = MeanAbsoluteError()

        if not config is None:
            self.gamma = config.get('BezierLossGamma', gamma)
            loss = config['BezierLossType']
            if loss == 'MAE':
                self.loss = MeanAbsoluteError()
            elif loss == 'MSE':
                self.loss = MeanSquaredError()
            elif loss == 'Huber':
                delta = self.config.get('HuberDelta', 1.0)
                self.loss = Huber(delta)
            else:
                raise NotImplementedError

    def call(self, y_true, y_pred):

        cp_true = tf.reshape(y_true, (-1, self.numOfControlPoints, self.dimentions))
        cp_hat = tf.reshape(y_pred, (-1, self.numOfControlPoints, self.dimentions))

        y_true_diff = cp_true[:, 1:, :] - cp_true[:, :-1, :]
        y_pred_diff = cp_hat[:, 1:, :] - cp_hat[:, :-1, :]
        
        cp_loss = self.loss(y_true, y_pred)
        cp_diff_loss = self.loss(y_true_diff, y_pred_diff)

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

        return self.gamma * cp_loss + (1-self.gamma) * cp_diff_loss


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

