import tensorflow as tf
from ColorModel import ClusterSuggestModel
import ColorData
import argparse
import yaml

NUM_CHANNELS = 3
num_epochs = 10000
def main():
    parser = argparse.ArgumentParser(description='Train Huereka Model')
    parser.add_argument('--config', type=str, default='cluster_config.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoint3')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    batch_size = config['data_params']['batch_size']

    model = ClusterSuggestModel(config)
    color_data = ColorData.ColorData(config)

    # clusters = color_data.compute_clusters_from_images()
    # return
    dataset, num_clusters = color_data.get_dataset_cluster(cluster_array_filename="clusters3.npy")
    data_train, data_test = color_data.get_dataset_batch(batch_size, dataset, num_epochs, num_clusters)
    data_train = data_train.make_one_shot_iterator().get_next()
    data_test = data_test.make_one_shot_iterator().get_next()

    input_cluster_num = config['data_params']['num_clusters'] - config['model_params']['num_colors']
    clusterPH = tf.compat.v1.placeholder(tf.float32, (batch_size,
        input_cluster_num, NUM_CHANNELS))
    labelsPH = tf.compat.v1.placeholder(tf.float32, (batch_size,
        config['model_params']['num_colors'], NUM_CHANNELS))

    model_out = model.build_model(clusterPH)
    model_out_clipped = tf.clip_by_value(model_out, 0., 1.)
    opt = model.define_train_op(model_out, labelsPH)

    save_steps = 20
    with tf.compat.v1.train.MonitoredTrainingSession(checkpoint_dir=args.checkpoint,
                                                     save_checkpoint_steps=save_steps) as sess:
        try:
            # for steps, lr in zip(config['training_params']['steps'], config['training_params']['lr']):
            for epoch in range(num_epochs):
                num_batches = int(num_clusters / batch_size)
                for i in range(num_batches):
                    batch = sess.run(data_train)

                    input_centers, removed_centers = color_data.remove_clusters(batch)
                    feed_dict = { clusterPH: input_centers, labelsPH: removed_centers}
                    _, loss = sess.run([model.train_op, model.loss], feed_dict=feed_dict)
                    if i % save_steps == 0:
                        print("### Epoch: " + str(epoch) + ", Iteration: " + str(i) + " ###")
                        train_loss = loss
                        batch = sess.run(data_test)
                        input_centers, removed_centers = color_data.remove_clusters(batch)
                        feed_dict = { clusterPH: input_centers, labelsPH: removed_centers}
                        predictions, loss = sess.run([model_out, model.loss], feed_dict=feed_dict)


                        print('{} {}'.format(train_loss, loss))

                    
        except KeyboardInterrupt:
            print('Stopping early due to keyboard interrupt')

if __name__ == '__main__':
    main()


