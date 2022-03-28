import unprocess
import tensorflow as tf

if __name__ == '__main__':
    # img = tf.io.read_file("../nerf_synthetic/lego/train/r_0.png")
    img = tf.io.read_file("test.png")
    png_file = tf.io.decode_png(img, channels=3)
    white_level = 255.0
    png_image = tf.cast(png_file, tf.float32) / white_level
    raw_img, metadata = unprocess.unprocess(png_image)
    raw_img_white_level = raw_img * 255.0
    tf.keras.utils.save_img('raw_test.png', raw_img_white_level.numpy(), scale=False)
    print(raw_img_white_level)
