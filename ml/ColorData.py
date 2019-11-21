import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import sys
from scipy.ndimage import gaussian_filter

sys.path.append('../clustering')
import ColorRemover

NUM_CHANNELS = 3

class ColorData:
    def __init__(self, config):
        self.data_params = config['data_params']
        self.model_params = config['model_params']

    def _decode_img(self, img):
        img = tf.io.read_file(img)
        img = tf.image.decode_image(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    def _reject_blank(self, img):
        return tf.logical_not(tf.reduce_all(
            tf.equal(tf.shape(img), tf.constant([343,600,3]))))

    def _resize(self, img):
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

    def get_dataset(self, test_size):
        dataset = tf.data.Dataset.list_files(self.data_params['img_folder'] + '/*')
        dataset = dataset.map(self._decode_img)
        dataset = dataset.filter(self._reject_blank)
        dataset = dataset.map(self._resize)
        dataset = dataset.map(self._random_crop_flip)

        batch_size = self.data_params['batch_size']
        data_test = dataset.take(test_size).repeat().shuffle(100)
        data_test = data_test.batch(batch_size).prefetch(8)
        data_train = dataset.skip(test_size).repeat().shuffle(100)
        data_train = data_train.batch(batch_size).prefetch(8)

        return data_train, data_test

    def remove_colors(self, batch, replacement_val):
        # return two batches of new images and lists of removed colors
        # also gaussian blurs the image

        # https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label
        # use this to only remove one connected component of each?
        new_img, removed_colors = [], []
        n_clusters = self.data_params['num_clusters']
        for img in batch:
            labels, img = ColorRemover.get_cluster_labels(img, n_clusters, 0)
            to_remove = np.random.choice(np.arange(n_clusters), 
                self.model_params['num_colors'], replace=False)
            centers = np.array([np.mean(img[labels==i], axis=0) for i in range(n_clusters)])
            removed = centers[to_remove]
            img = ColorRemover.remove_colors(img, labels, to_remove, replacement_val)
            img = gaussian_filter(img, sigma=.5)
            new_img.append(img)
            removed_colors.append(removed)
        return np.array(new_img, dtype=np.float32), np.array(removed_colors, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description='Art Color Dataset')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    tf.compat.v1.enable_eager_execution()

    color_data = ColorData(config)
    data_train, data_test = color_data.get_dataset(1000)

    # for weeding out images that break when decoded
    # for x in data_train:
    #     print(x.numpy())
    #     img = tf.io.read_file(x[0])
    #     img = tf.image.decode_image(img)
    #     img.numpy()

    for batch in data_train:
        i=0
        img, colors = color_data.remove_colors(batch, replacement_val=1.)
        for img, colors in zip(img, colors):
            plt.figure()
            plt.axis('off')
            plt.imshow(batch[i])
            i += 1
            print(colors)
            plt.figure()
            plt.axis('off')
            plt.imshow(img)
            plt.figure()
            plt.axis('off')
            x_vals = range(0, len(colors))
            height = 10
            plt.bar(x_vals, height, color=colors)
            plt.show()
            

if __name__ == '__main__':
    main()