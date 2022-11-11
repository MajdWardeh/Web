from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError


class ControlPointsMetric(tf.keras.metrics.Metric):

  def __init__(self, name='cp', numOfCp=5, dimention=3, type='MSE', **kwargs):
    super(ControlPointsMetric, self).__init__(name=name, **kwargs)
    self.numOfControlPoints = numOfCp
    self.dimentions = dimention
    self.metricList = []

    metricsNum = 3
    if type == 'MSE':
        self.metricList = [MeanSquaredError() for i in range(metricsNum)]
    elif type == 'MAE':
        self.metricList = [MeanAbsoluteError() for i in range(metricsNum)]
    else:
        raise RuntimeError('Only MSE or MAE are available curently')

  def update_state(self, y_true, y_pred, sample_weight=None):
    cp_true = tf.reshape(y_true, (-1, self.numOfControlPoints, self.dimentions))
    cp_hat = tf.reshape(y_pred, (-1, self.numOfControlPoints, self.dimentions))
    self.metricList[0].update_state(cp_true, cp_hat, sample_weight)
    self.metricList[1].update_state(cp_true[:, -1, :], cp_hat[:, -1, :], sample_weight)
    

    y_true_diff = cp_true[:, 1:, :] - cp_true[:, :-1, :]
    y_pred_diff = cp_hat[:, 1:, :] - cp_hat[:, :-1, :]

    self.metricList[2].update_state(y_true_diff, y_pred_diff)

    # yaw_cp_true = y_true[:, ] 

    # values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
    # values = tf.cast(values, self.dtype)
    # if sample_weight is not None:
    #   sample_weight = tf.cast(sample_weight, self.dtype)
    #   sample_weight = tf.broadcast_to(sample_weight, values.shape)
    #   values = tf.multiply(values, sample_weight)
    # self.true_positives.assign_add(tf.reduce_sum(values))

  def result(self):
    return [metric.result() for metric in self.metricList]