from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial import distance
import argparse
from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from colormap import rgb2hex
import math 

def cluster_colors_kmeans(sampled_colors, n_clusters):

    # using KMeans algorithm to find n clusters
    kmeans = KMeans(n_clusters=n_clusters).fit(sampled_colors)
    
    # reconstructing compressed image with the found clusters
    
    centers = kmeans.cluster_centers_
    plot_colors(centers)
    return (centers* 255).astype(int)

def compute_squared_dist_matrix(points):
    dist_matrix = np.sum(((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2 * (3/2)), axis = -1)

    return dist_matrix


def compute_exp_dist_matrix(points):
    dist_matrix = np.sum(np.sqrt((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2), axis = -1)
    exp_dist = np.exp(dist_matrix) - 1.0
    return exp_dist

def get_sampled_colors(ims, samples_per_image=1000):
    sampled_colors = []

    for im in ims:
        im_numpy = np.array(im)
        if im_numpy.dtype == np.uint8:
            im_numpy = im_numpy.astype(np.float64)/255        
        shape = im_numpy.shape
        im_X = im_numpy.reshape(shape[0]*shape[1], 3)
        im_X_to_sample = im_X.copy()
        np.random.shuffle(im_X_to_sample)
        im_X_sampled = im_X_to_sample[:samples_per_image]
        sampled_colors.append(im_X_sampled)
    
    sampled_colors = np.array(sampled_colors)
    sampled_colors = sampled_colors.reshape((sampled_colors.shape[1], -1))
    return sampled_colors

def get_cluster_centers(colors, labels, n_clusters):
    centers = [np.mean(colors[labels == i], axis=0) for i in range(n_clusters)]
    return centers

def cluster_colors(colors, n_clusters):
    # using DBSCAN algorithm to find n clusters
    # dist_matrix = compute_exp_dist_matrix(colors)
    dist_matrix = compute_exp_dist_matrix(colors)

    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage='average')
    
    clustering.fit(dist_matrix)
    labels = clustering.labels_
    return labels

    # reconstructing compressed image with the found clusters
    
    # centers = ward.cluster_centers_
    # plot_colors(centers)
    # return (centers* 255).astype(int)

def plot_colors(colors):
    x_vals = range(0, len(colors))
    height = 10
    plt.bar(x_vals, height, color=colors)
    plt.show()

def floatRGB2hex(color):
    """
    Convert the input color (RGB) to hex
    """
    R = min(256, math.floor(color[0] * 256)) 
    G = min(256, math.floor(color[1] * 256)) 
    B = min(256, math.floor(color[2] * 256)) 
    return rgb2hex(R, G, B)

def load_images(image_dir):
    files = [f for f in listdir(image_dir) if (isfile(join(image_dir, f)) and f != '.DS_Store')]
    images = []
    for file in files:
        path = image_dir + "/" + file
        im = Image.open(path)
        images.append(im)
    return images

def compute_clusters(colors, num_clusters):
    labels = cluster_colors(colors, num_clusters)
    centers = get_cluster_centers(colors, labels, num_clusters)
    return labels, centers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--num_clusters", type=int, default=10)
    parser.add_argument("--samples_per_image", type=int, default=1000)
    args = parser.parse_args()

    image_dir = args.image_dir
    images = load_images(image_dir)

    num_clusters = args.num_clusters
    num_samples = args.samples_per_image

    colors = get_sampled_colors(images, num_samples)
    labels, centers = compute_clusters(colors, num_clusters)
    return labels, centers
        
if __name__ == "__main__":
    main()