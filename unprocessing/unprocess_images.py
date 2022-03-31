import os
import unprocess
import tensorflow as tf

if __name__ == '__main__':
    for filename in os.listdir('../data/nerf_synthetic/lego/test'):
        img = tf.io.read_file(filename)
        png_file = tf.io.decode_png(img, channels=3)
        white_level = 255.0
        png_image = tf.cast(png_file, tf.float32) / white_level
        raw_img, metadata = unprocess.unprocess(png_image)
        raw_img_white_level = raw_img * 255.0
        raw_numpy_img = raw_img_white_level.numpy()
        raw_img_noise = unprocess.add_noise(raw_numpy_img)
        tf.keras.utils.save_img('../data/nerf_raw_noise/lego/test/' + filename, raw_img_noise, scale=False)
    for filename in os.listdir('../data/nerf_synthetic/lego/train'):
        img = tf.io.read_file(filename)
        png_file = tf.io.decode_png(img, channels=3)
        white_level = 255.0
        png_image = tf.cast(png_file, tf.float32) / white_level
        raw_img, metadata = unprocess.unprocess(png_image)
        raw_img_white_level = raw_img * 255.0
        raw_numpy_img = raw_img_white_level.numpy()
        raw_img_noise = unprocess.add_noise(raw_numpy_img)
        tf.keras.utils.save_img('../data/nerf_raw_noise/lego/train/' + filename, raw_img_noise, scale=False)
