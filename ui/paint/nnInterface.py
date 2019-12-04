import sys 
sys.path.append("../../ml")
import tensorflow as tf
import numpy as np
import yaml
import ColorModel
import cv2

NUM_CHANNELS = 3
class NNInterface:
    def __init__(self, config, checkpoint):
        self.sess = tf.Session()
        with open(config) as f:
            config = yaml.load(f)
        self.img_size = config['data_params']['img_size']
        self.imagePH = tf.compat.v1.placeholder(tf.float32, (config['data_params']['batch_size'],
            config['data_params']['img_size'], config['data_params']['img_size'], NUM_CHANNELS))
        model = ColorModel.ColorSuggestModelSeparate(config)
        model_out = model.buildModel(self.imagePH)
        self.model_out_clipped = tf.clip_by_value(model_out, 0., 1.)
        

        saver = tf.compat.v1.train.Saver()
        checkpoint = tf.train.latest_checkpoint(checkpoint)
        saver.restore(self.sess, checkpoint)


    def __del__(self):
        if self.sess:
            self.sess.close()

    def predict(self, img):
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img[np.newaxis]
        feed_dict = { self.imagePH:img }
        colors_out = self.sess.run(self.model_out_clipped, feed_dict=feed_dict)
        color = colors_out[0][0]
        return [int(x * 255) for x in color]


nn = NNInterface("../../ml/config.yaml", "../checkpoint")
img = cv2.imread("input/removed_color.png")
pred = nn.predict(img)
print(pred)