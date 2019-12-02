import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml

import ColorModel
import ColorData

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

def main():
    """
    Test the color predcting model
    """
    parser = argparse.ArgumentParser(description='Train Huereka Model')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoint')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    # prepare data in
    color_data = ColorData.ColorData(config)
    # small train set: 4
    # normal train set: 670
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
    model_out_clipped = tf.clip_by_value(model_out, 0., 1.)
    opt = model.buildOpt(model_out, labelsPH)
    
    # loop
    with tf.Session() as sess:
        saver = tf.compat.v1.train.Saver()
        checkpoint = tf.train.latest_checkpoint(args.checkpoint)
        saver.restore(sess, checkpoint)
        try:
            while True:
                batch = sess.run(data_test)
                img, colors = color_data.remove_colors(batch, replacement_val=1.)
                feed_dict = { imagePH:img, labelsPH:colors }
                colors_out, loss = sess.run([model_out_clipped, opt['loss']], feed_dict=feed_dict)

                print(loss)

                for i in range(len(batch)):
                    print('    {}'.format([int(x * 255) for x in colors[i][0]]))
                    print('    {}'.format([int(x * 255) for x in colors_out[i][0]]))
                    displayOne(batch, img, colors_out, i)

        except KeyboardInterrupt:
            print('Stopping early due to keyboard interrupt')



if __name__ == '__main__':
    main()