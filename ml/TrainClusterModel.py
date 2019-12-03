import tensorflow as tf
from ColorModel import ClusterSuggestModel
import ColorData
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser(description='Train Huereka Model')
    parser.add_argument('--config', type=str, default='cluster_config.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoint')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    model = ClusterSuggestModel(config)
    color_data = ColorData.ColorData(config)
    data_train, data_test = color_data.get_dataset_clusters(670)
    data_train = data_train.make_one_shot_iterator().get_next()
    data_test = data_test.make_one_shot_iterator().get_next()

    clusterPH = tf.compat.v1.placeholder(tf.float32, (config['data_params']['batch_size'],
        config['data_params']['num_clusters'], NUM_CHANNELS))
    labelsPH = tf.compat.v1.placeholder(tf.float32, (config['data_params']['batch_size'],
        config['model_params']['num_colors'], NUM_CHANNELS))

    model_out = model.buildModel(clusterPH)
    opt = model.buildOpt(model_out, labelsPH)

    with tf.compat.v1.train.MonitoredTrainingSession(checkpoint_dir=args.checkpoint,
                                                    save_checkpoint_steps=save_steps) as sess:
        try:
            for steps, lr in zip(config['training_params']['steps'], config['training_params']['lr']):
                for i in range(steps):
                    batch = sess.run(data_train)
                    input_centers, removed_centers = color_data.remove_clusters(batch)
                    feed_dict = { clusterPH: input_centers, labelsPH: removed_centers}
                    _, loss = sess.run([model.loss, model.train_op], feed_dict=feed_dict)

                    if i % save_steps == 0:
                        print("### Iteration: " + str(i) + " ###")
                        train_loss = loss
                        batch = sess.run(data_test)
                        input_centers, removed_centers = color_data.remove_clusters(batch)
                        feed_dict = { clusterPH: input_centers, labelsPH: removed_centers}
                        predictions, loss = sess.run([model_out, model.loss], feed_dict=feed_dict)

                        summary, step = sess.run([write_op, opt['global_step']], { loss_var:loss })
                        test_writer.add_summary(summary, step)
                        test_writer.flush()

                        print('{} {} {}'.format(train_loss, loss))
                        print('    {}'.format([int(x * 255) for x in predictions[0]]))
                    
        except KeyboardInterrupt:
            print('Stopping early due to keyboard interrupt')

if __name__ == '__main__':
    main()


