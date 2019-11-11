from MoodboardColorPicker import cluster_colors, load_images
import numpy as np
import sys
import os
import argparse
from skimage.transform import resize
import matplotlib.pyplot as plt


def get_image_colors(im):
	colors = np.array(im)
	if colors.dtype == np.uint8:
		colors = colors.astype(np.float64) / 255
	return colors

def get_cluster_labels(image, num_clusters):
	colors = get_image_colors(image)
	original_shape = colors.shape
	colors = colors.reshape((colors.shape[0] * colors.shape[1], 3))
	cluster_labels = cluster_colors(colors, num_clusters)
	cluster_labels = cluster_labels.reshape((original_shape[0], original_shape[1]))
	return cluster_labels, colors

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--image_dir", type=str)
	parser.add_argument("--num_clusters", type=int, default=10)
	args = parser.parse_args()

	image_dir = args.image_dir
	num_clusters = args.num_clusters

	images = load_images(image_dir)
	for image in images:
		labels, colors = get_cluster_labels(image, num_clusters)
		remove_colors(colors, labels, [1, 2])

if __name__ == "__main__":
	main()