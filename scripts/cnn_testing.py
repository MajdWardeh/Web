import cv2
import tensorflow as tf
import numpy as np
from nets import resnet8 as prediction_network
import sys
import gflags
from ddr_learner.common_flags import FLAGS

class TestNetwork():

    def __init__(self):
        pass


    def build_test_graph(self):
        """This graph will be used for testing. In particular, it will
           compute the loss on a testing set, or for prediction of trajectories.
        """
        image_height, image_width = self.config.test_img_height, \
                                    self.config.test_img_width

        self.num_channels = 3
        input_uint8 = tf.placeholder(tf.uint8, [None, image_height,
                                    image_width, self.num_channels],
                                    name='raw_input')


        input_mc = self.preprocess_image(input_uint8)

        pnt_batch = tf.placeholder(tf.float32, [None, self.config.output_dim],
                                          name='gt_labels')


        with tf.name_scope("trajectory_prediction"):
            pred_pnt, convs_layer_output = prediction_network(input_mc,
                    output_dim=self.config.output_dim, f=self.config.f)

        with tf.name_scope("compute_loss"):
            point_loss = tf.losses.mean_squared_error(labels=pnt_batch[:,:2],
                                                      predictions=pred_pnt[:,:2])

            vel_loss = tf.losses.mean_squared_error(labels=pnt_batch[:, 2],
                                                    predictions=pred_pnt[:, 2])
            total_loss = point_loss + vel_loss


        with tf.name_scope("metrics"):
            _, var = tf.nn.moments(pred_pnt, axes=-1)
            std = tf.sqrt(var)

        self.inputs_img = input_uint8
        self.pred_pnt = pred_pnt
        self.gt_pnt = pnt_batch
        self.pred_stds = std
        self.point_loss = point_loss
        self.total_loss = total_loss
        self.vel_loss = vel_loss

        self.convs_layer_output = convs_layer_output

    def preprocess_image(self, image):
        """ Preprocess an input image
        Args:
            Image: A uint8 tensor
        Returns:
            image: A preprocessed float32 tensor.
        """
        image = tf.image.resize_images(image,
                [self.config.img_height, self.config.img_width])
        image = tf.cast(image, dtype=tf.float32)
        image = tf.divide(image, 255.0)
        return image

    def setup_inference(self, config, mode, sess):
        """Sets up the inference graph.
        Args:
            mode: either 'loss' or 'prediction'. When 'loss', it will be used for
            computing a loss (gt trajectories should be provided). When
            'prediction', it will just make predictions (to be used in simulator)
            config: config dictionary. it should have target size and trajectories
        """
        self.mode = mode
        self.config = config
        self.build_test_graph()

        print("hellllllllllllllllllllo")
        for var in tf.trainable_variables():
            print(var)
        self.saver = tf.train.Saver([var for var in tf.trainable_variables()])
        
        self.saver.restore(sess, self.config.ckpt_file)

    def inference(self, inputs, sess):
        results = {}
        fetches = {}
        if self.mode == 'loss':
            fetches["vel_loss"] = self.vel_loss
            fetches["pnt_loss"] = self.point_loss
            fetches["stds"] = self.pred_stds

            results = sess.run(fetches,
                               feed_dict= {self.inputs_img: inputs['images'],
                                           self.gt_pnt: inputs['gt_labels']})
        if self.mode == 'prediction':
            results['predictions'] = sess.run(self.pred_pnt, feed_dict = {
                self.inputs_img: inputs['images']})
        return results

    def inference_convs(self, inputs, sess):
        results = {}
        results['predictions'] = sess.run(self.convs_layer_output, feed_dict={
                self.inputs_img: inputs['images']})
        return results

    def save_model(self, ckpt_file, sess):
        self.saver.save(sess, ckpt_file)

def run_network():
    with tf.Session() as sess:
        # FLAGS.ckpt_file = "~/drone_racing_ws/catkin_ddr/src/sim2real_drone_racing/learning/deep_drone_racing_learner/src/ddr_learner/results/best_model/navigation_model"
        FLAGS.ckpt_file = "./model/navigation_model"
        FLAGS.f = 0.5
        testNetwork = TestNetwork()

        sess.run(tf.global_variables_initializer())

        testNetwork.setup_inference(FLAGS, mode='prediction', sess=sess)

        inputs = {}
        cv_input_image = cv2.imread("../images/image1.jpg")
        cv2.imshow("image", cv_input_image)
        cv2.waitKey(0)
        cv_input_image = cv2.resize(cv_input_image, (300, 200),
                                 interpolation=cv2.INTER_LINEAR)
        inputs['images'] = cv_input_image[None]
        output = testNetwork.inference(inputs, sess)
        print(output)

        convsOutput = testNetwork.inference_convs(inputs, sess)
        print(convsOutput)
        # testNetwork.save_model('./model2/navigation_model2', sess)

def parse_flags(argv):
    # Utility main to load flags
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
      print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
      sys.exit(1)

if __name__ == "__main__":
    parse_flags(sys.argv)
    run_network()