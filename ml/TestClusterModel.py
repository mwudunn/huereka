import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml

from ColorModel import ClusterSuggestModel
import ColorData


NUM_CHANNELS = 3
num_epochs = 10

def display_clusters(batch, input_centers, output_centers, removed_centers, i=0):
    height = 10
    input_colors = input_centers[i]
    output_colors = output_centers[i]
    removed_colors = removed_centers[i]

    x_inputs = range(0, len(input_colors))
    x_outputs = range(0, len(output_colors))
    x_removed = range(0, len(removed_colors))

    plt.subplot(3,1,1)
    plt.bar(x_inputs, height, color=input_colors)

   
    plt.subplot(3,1,2)
    plt.bar(x_outputs, height, color=output_colors)

    plt.subplot(3,1,3)
    plt.bar(x_removed, height, color=removed_colors)
    plt.tight_layout()
    plt.show()

def main():
    """
    Test the color predcting model
    """
    parser = argparse.ArgumentParser(description='Test Huereka Cluster Model')
    parser.add_argument('--config', type=str, default='cluster_config.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoint3')
    parser.add_argument('--file', type=str, default='clusters3.npy')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    batch_size = config['data_params']['batch_size']

    model = ClusterSuggestModel(config)
    color_data = ColorData.ColorData(config)

    dataset, num_clusters = color_data.get_dataset_cluster(cluster_array_filename=args.file)
    data_train, data_test = color_data.get_dataset_batch(batch_size, dataset, num_epochs, num_clusters)
    data_train = data_train.make_one_shot_iterator().get_next()
    data_test = data_test.make_one_shot_iterator().get_next()

    input_cluster_num = config['data_params']['num_clusters'] - config['model_params']['num_colors']
    clusterPH = tf.compat.v1.placeholder(tf.float32, (batch_size,
        input_cluster_num, NUM_CHANNELS))
    labelsPH = tf.compat.v1.placeholder(tf.float32, (batch_size,
        config['model_params']['num_colors'], NUM_CHANNELS))

    model_out = model.build_model(clusterPH)
    opt = model.define_train_op(model_out, labelsPH)
    
    # loop
    with tf.Session() as sess:
        saver = tf.compat.v1.train.Saver()
        checkpoint = tf.train.latest_checkpoint(args.checkpoint)
        saver.restore(sess, checkpoint)
        try:
            while True:
                batch = sess.run(data_test)
                input_centers, removed_centers = color_data.remove_clusters(batch)
                feed_dict = { clusterPH: input_centers, labelsPH: removed_centers}
                clusters_out, loss = sess.run([model_out, model.loss], feed_dict=feed_dict)

                print(loss)

                display_clusters(batch, input_centers, clusters_out, removed_centers)

        except KeyboardInterrupt:
            print('Stopping early due to keyboard interrupt')



if __name__ == '__main__':
    main()