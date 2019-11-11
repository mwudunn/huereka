import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import sqlite3
import pandas as pd
import requests

NUM_CHANNELS = 3
STYLES = {
    'watercolor': 'media_watercolor',
    'vectorart': 'media_vectorart'
}

class ColorData:
    def __init__(self, config):
        self.data_params = config['data_params']

    def _decode_img(self, img):
        img = tf.image.decode_image(img)
        img = tf.image.convert_image_dtype(img, tf.float32)

        # resize to crop_pad_px larger than the final img will be
        base_size = self.data_params['img_size']
        crop_pad_px = self.data_params['crop_pad_px']
        img_shape = [base_size + crop_pad_px] * 2
        img = tf.expand_dims(img, 0)
        img = tf.compat.v1.image.resize_bilinear(img, img_shape)
        return img[0]

    def _random_crop_flip(self, img):
        crop_size = self.data_params['img_size']
        img = tf.image.random_crop(img, [crop_size, crop_size, NUM_CHANNELS])
        img = tf.image.random_flip_left_right(img)

        return img

    def _generator(self):
        db = sqlite3.connect(self.data_params['database'])
        query = """SELECT m.src FROM modules AS m, automatic_labels AS a
                   WHERE m.mid == a.mid AND a.{0} == 'positive'""".format(self.style)
        df = pd.read_sql(query, db)
        for url in df['src']:
            yield requests.get(url).content


    def get_dataset(self, style):
        self.style = STYLES[style]
        dataset = tf.data.Dataset.from_generator(self._generator, tf.string)
        dataset = dataset.map(self._decode_img)
        dataset = dataset.map(self._random_crop_flip)

        batch_size = self.data_params['batch_size']
        dataset = dataset.shuffle(100)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(8)

        return dataset



def main():
    parser = argparse.ArgumentParser(description='Art Color Dataset')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    tf.compat.v1.enable_eager_execution()

    color_data = ColorData(config)
    dataset = color_data.get_dataset(config['data_params']['style'])

    for batch in dataset.take(1):
        for img in batch:
            plt.figure()
            plt.axis('off')
            plt.imshow(img.numpy())
            plt.show()


if __name__ == '__main__':
    main()