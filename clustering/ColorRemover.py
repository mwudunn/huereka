from MoodboardColorPicker import cluster_colors, load_images
import numpy as np
import sys
import os
import argparse
from skimage.transform import resize
import matplotlib.pyplot as plt


def get_image_colors(im, rescale_size):
	numpy_im = np.array(im)
	colors = resize(numpy_im, (rescale_size, rescale_size), anti_aliasing=True)
	if colors.dtype == np.uint8:
		colors = colors.astype(np.float64) / 255
	return colors

def get_cluster_labels(image, num_clusters, rescale_size):
	colors = get_image_colors(image, rescale_size)
	original_shape = colors.shape
	colors = colors.reshape((colors.shape[0] * colors.shape[1], 3))
	cluster_labels = cluster_colors(colors, num_clusters)
	cluster_labels = cluster_labels.reshape((original_shape[0], original_shape[1]))
	return cluster_labels, colors

def remove_colors(image, labels, to_remove):
    num = np.max(labels)
    new_image = image.copy()
    labels_sorted = np.asarray(to_remove + [x for x in labels if x not in np.arange(10)])
    labels_map = np.argsort(labels_sorted - 1)
    new_labels = labels_map[labels - 1]
    new_image[new_labels < len(to_remove)] = 0.
    return new_image

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--image_dir", type=str)
	parser.add_argument("--num_clusters", type=int, default=10)
	parser.add_argument("--rescale_size", type=int, default=128)
	args = parser.parse_args()

	image_dir = args.image_dir
	num_clusters = args.num_clusters
	rescale_size = args.rescale_size

	images = load_images(args.image_dir)
	for image in images:
		labels, colors = get_cluster_labels(image, args.num_clusters, rescale_size)
		remove_colors(colors, labels, [1, 2])

if __name__ == "__main__":
    main()