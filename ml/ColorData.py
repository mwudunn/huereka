import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import sys
from scipy.ndimage import gaussian_filter
import glob
import cv2

sys.path.append('../clustering')
import ColorRemover
from MoodboardColorPicker import get_cluster_centers


NUM_CHANNELS = 3

def displayOne(batch, img, colors, i=0):
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.imshow(batch[i])
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(img[i])
    plt.subplot(1, 3, 3)
    plt.axis('off')
    x_vals = range(0, len(colors[i]))
    plt.bar(x_vals, 10, color=colors[i])
    plt.show()

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
        dataset = tf.data.Dataset.list_files(self.data_params['img_folder'] + '/*', shuffle=False)
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

    def get_dataset_cluster(self, cluster_array_filename="clusters2.npy"):
        try:
            clusters_list = np.load(cluster_array_filename, allow_pickle=True).astype(np.float32)
        except:
            clusters_list = compute_clusters_from_images(self.data_params['img_folder'], self.data_params['img_size'], self.data_params['num_clusters'], cluster_array_filename)
        
        num_clusters = clusters_list.shape[0]
        dataset = tf.data.Dataset.from_tensor_slices(clusters_list)
        return dataset, num_clusters

    def get_dataset_batch(self, test_size, dataset, num_epochs, num_clusters):
        
        batch_size = self.data_params['batch_size']
        repeat_amount = int(num_epochs * num_clusters / batch_size)
        dataset = dataset.shuffle(10 * batch_size)
        data_test = dataset.take(test_size)
        data_test = data_test.repeat(repeat_amount).batch(batch_size)
        data_train = dataset.skip(test_size)
        data_train = data_train.repeat(repeat_amount).batch(batch_size)


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

        removed_colors = np.sort(np.array(removed_colors), axis=0)
        return np.array(new_img, dtype=np.float32), removed_colors

    def remove_clusters(self, batch):
        num_colors = self.model_params['num_colors']
        input_centers, removed_centers = [], []
        for clusters in batch:
            np.random.shuffle(clusters)
            to_remove, to_keep = clusters[:num_colors], clusters[num_colors:]
            to_remove = np.sort(to_remove, axis=0)
            input_centers.append(to_keep)
            removed_centers.append(to_remove)
        return np.array(input_centers, dtype=np.float32), np.array(removed_centers, dtype=np.float32)



    def _gaussian_blur(self, batch):
        pass

def get_image_clusters(img, n_clusters):
    labels, img = ColorRemover.get_cluster_labels(img, n_clusters, 0)
    clusters = get_cluster_centers(img, labels, n_clusters)
    return clusters

def compute_clusters_from_images(img_folder, img_size, n_clusters, cluster_array_filename):
    clusters_list = []
    i = 0
    for filename in glob.iglob(img_folder + '/*.jpg'):
        im = cv2.imread(filename)
        if np.size(im) <= 1:
            continue
        im = cv2.resize(im, (img_size, img_size))
        clusters = get_image_clusters(im, n_clusters)
        clusters_list.append(clusters)
        i += 1
        if i % 100 == 0:
            print("Images Processed: " + str(i + 1))
    print("Done...")
    clusters_list = np.array(clusters_list)
    np.save(cluster_array_filename, clusters_list)
    return clusters_list

def filter_corrupt_images(img_folder):
    import os
    filtered_dataset = []
    i = 0
    for filename in glob.iglob(img_folder + '/*.jpg'):
        
        img = tf.io.read_file(filename)
        
        try:
            img = tf.image.decode_image(img)
            img.numpy()
            if img.shape == (343,600,3):
                print("Image Deleted.")
                os.remove(filename)            
        except:
            print("Image deleted.")
            os.remove(filename)
        if i % 100 == 0:
            print(i)
        i+=1

def main():
    parser = argparse.ArgumentParser(description='Art Color Dataset')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    tf.compat.v1.enable_eager_execution()

    color_data = ColorData(config)
    img_folder = config['data_params']['img_folder']

    # for weeding out images that break when decoded
    filtered_data_train = filter_corrupt_images(img_folder)

    # for batch in filtered_data_test:
    #     img, colors = color_data.remove_colors(batch, replacement_val=1.)
    #     for i in range(len(batch)):
    #         displayOne(batch, img, colors, i)
            

if __name__ == '__main__':
    main()