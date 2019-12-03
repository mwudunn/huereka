import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml

import ColorModel
import ColorData

NUM_CHANNELS = 3

def main():
    """
    Train the color predcting model
    """
    parser = argparse.ArgumentParser(description='Train Huereka Model')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoint')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    # prepare data in
    color_data = ColorData.ColorData(config)
    data_train, data_test = color_data.get_dataset(670)
    data_train = data_train.make_one_shot_iterator().get_next()
    data_test = data_test.make_one_shot_iterator().get_next()

    # prepare model and optimization
    imagePH = tf.compat.v1.placeholder(tf.float32, (config['data_params']['batch_size'],
        config['data_params']['img_size'], config['data_params']['img_size'], NUM_CHANNELS))
    labelsPH = tf.compat.v1.placeholder(tf.float32, (config['data_params']['batch_size'],
        config['model_params']['num_colors'], NUM_CHANNELS))

    model = ColorModel.ColorSuggestModelSeparate(config)
    model_out = model.buildModel(imagePH)
    opt = model.buildOpt(model_out, labelsPH)

    # summary for tensorboard
    train_writer = tf.compat.v1.summary.FileWriter(args.checkpoint + '/logs/train')
    test_writer = tf.compat.v1.summary.FileWriter(args.checkpoint + '/logs/test')
    loss_var = tf.Variable(0.0)
    tf.summary.scalar("loss", loss_var)
    write_op = tf.compat.v1.summary.merge_all()
    
    # loop
    save_steps = 10
    with tf.compat.v1.train.MonitoredTrainingSession(checkpoint_dir=args.checkpoint,
                                                     save_checkpoint_steps=save_steps) as sess:
        try:
            for steps, lr in zip(config['training_params']['steps'], config['training_params']['lr']):
                sess.run(opt['lr_assign'], feed_dict={ opt['lrPH']:lr })
                for i in range(steps):
                    batch = sess.run(data_train)
                    img, colors = color_data.remove_colors(batch, replacement_val=1.)
                    feed_dict = { imagePH:img, labelsPH:colors }
                    _, loss = sess.run([opt['opt'], opt['loss']], feed_dict=feed_dict)

                    summary, step = sess.run([write_op, opt['global_step']], { loss_var:loss })
                    train_writer.add_summary(summary, step)
                    train_writer.flush()

                    if i % save_steps == 0:
                        print("### Iteration: " + str(i) + " ###")
                        train_loss = loss
                        batch = sess.run(data_test)
                        img, colors = color_data.remove_colors(batch, replacement_val=1.)
                        feed_dict = { imagePH:img, labelsPH:colors }
                        colors_out, loss = sess.run([model_out, opt['loss']], feed_dict=feed_dict)

                        summary, step = sess.run([write_op, opt['global_step']], { loss_var:loss })
                        test_writer.add_summary(summary, step)
                        test_writer.flush()

                        print('{} {} {}'.format(step, train_loss, loss))
                        print('    {}'.format([int(x * 255) for x in colors_out[0][0]]))

        except KeyboardInterrupt:
            print('Stopping early due to keyboard interrupt')



if __name__ == '__main__':
    main()