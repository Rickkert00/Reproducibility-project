import os
import tensorflow as tf

from unprocessing import unprocess

metadata = {'cam2rgb': tf.constant([[1.6148667, -0.5449992, -0.06986756], [-0.19303098,  1.5009984, -0.3079673], [0.02317855, -0.43180132,  1.4086229]], shape=(3,3), dtype=tf.float32),
            'rgb_gain': tf.constant(1.1466613, shape=(), dtype=tf.float32),
            'red_gain': tf.constant(2.0274398, shape=(), dtype=tf.float32),
            'blue_gain': tf.constant(1.6323329, shape=(), dtype=tf.float32)}

if __name__ == '__main__':
    for filename in os.listdir('../data/our_nerf_synthetic/lego/train'):
        img = tf.io.read_file('../data/our_nerf_synthetic/lego/train/' + filename)
        png_file = tf.io.decode_png(img, channels=3)
        white_level = 255.0
        png_image = tf.cast(png_file, tf.float32) / white_level
        raw_img = unprocess.unprocess(png_image, metadata)
        raw_img_white_level = raw_img * 255.0
        raw_numpy_img = raw_img_white_level.numpy()
        # raw_numpy_img = unprocess.add_noise(raw_numpy_img)
        tf.keras.utils.save_img('../data/nerf_raw/lego/train/' + filename, raw_numpy_img, scale=False)


    # for filename in os.listdir('../data/nerf_synthetic/lego/test'):
    #     img = tf.io.read_file('../data/nerf_synthetic/lego/test/' + filename)
    #     # png_file = tf.io.decode_png(img, channels=3)
    #     # white_level = 255.0
    #     # png_image = tf.cast(png_file, tf.float32) / white_level
    #     # raw_img, metadata = unprocess.unprocess(png_image)
    #     # raw_img_white_level = raw_img * 255.0
    #     # raw_numpy_img = raw_img_white_level.numpy()
    #     # raw_img_noise = unprocess.add_noise(raw_numpy_img)
    #     tf.keras.utils.save_img('../data/nerf_raw_noise/lego/test/' + filename, img, scale=False)
