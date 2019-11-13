import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import ColorModel
import ColorData


def main():
	"""
	Train the color predcting model
	"""
	parser = argparse.ArgumentParser(description='Train Huereka Model')
	parser.add_argument('--config', type=str, default='config.yaml')
	args = parser.parse_args()

	with open(args.config) as f:
	    config = yaml.load(f, Loader=yaml.CLoader)

	tf.compat.v1.enable_eager_execution()

	color_data = ColorData.ColorData(config)
	dataset = color_data.get_dataset(config['data_params']['style'])




if __name__ == '__main__':
	main()