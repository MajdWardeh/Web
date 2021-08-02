import tensorflow as tf
from tensorflow.keras.losses import Loss, MeanAbsoluteError
from tensorflow.keras import backend as k
from scipy.special import binom

class BezierLoss(Loss):

    def __init__(self, n=4, acc=100, gamma=0.6):
        # n is the order of the Bezier.
        super().__init__()
        self.gamma = gamma
        self.A = np.zeros((acc, n+1))
        self.mae = MeanAbsoluteError()
        t_space = np.linspace(0, 1, acc)
        for i, ti in enumerate(t_space):
            for j in range(n+1):
                self.A[i, j] = binom(n, j) * math.pow(1-ti, n-j) * math.pow(ti, j) 
        self.A = tf.convert_to_tensor(self.A, dtype=tf.float32)

    def call(self, y_true, y_pred):
        bs = tf.shape(y_true)[0]
        cp_true = tf.reshape(y_true, (bs, 4, 3))
        cp_true = tf.concat([tf.zeros(shape=(bs, 1, 3)), cp_true], axis=1)
        cp_pred = tf.reshape(y_pred, (bs, 4, 3))
        cp_pred = tf.concat([tf.zeros(shape=(bs, 1, 3)), cp_pred], axis=1)

        P_true = tf.matmul(self.A, cp_true)
        P_pred = tf.matmul(self.A, cp_pred)

        MAE_curve = self.mae(P_true, P_pred)
        MAE_cp = self.mae(y_true, y_pred)
        return self.gamma * MAE_curve + (1-self.gamma) * MAE_cp

        # abs_error = tf.math.abs(P_true-P_pred)
        # MAExyz = tf.reduce_mean(abs_error, 0)
        # MAE = tf.reduce_mean(MAExyz)
        # return MAE


def main():
    pass

if __name__ == '__main__':
    main()

