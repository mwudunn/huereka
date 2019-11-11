import sys 
import numpy as np 
sys.path.append("../../clustering")
from MoodboardColorPicker import *


def image_load_test():
    images = load_images("input")
    num_clusters = 10
    num_samples = 1000
    colors = get_sampled_colors(images, num_samples)
    labels, centers = compute_clusters(colors, num_clusters)
    print(centers)

