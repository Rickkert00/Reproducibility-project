from copy import copy

import tensorflow as tf
import json

import unprocess


def unprocess_image(image_name):
    # img = tf.io.read_file("../nerf_synthetic/lego/train/r_0.png")
    img = tf.io.read_file(f"../data/nerf_synthetic/lego/test/{image_name}.png")
    png_file = tf.io.decode_png(img, channels=3)
    white_level = 255.0
    png_image = tf.cast(png_file, tf.float32) / white_level
    raw_img, metadata = unprocess.unprocess(png_image)
    raw_img_white_level = raw_img * 255.0
    tf.keras.utils.save_img(f'../data/nerf_raw/lego/test/{image_name}.png', raw_img_white_level.numpy(), scale=False)


if __name__ == '__main__':
    NUM_IMAGES = 120
    TOTAL_IMAGES = 200
    ratio = NUM_IMAGES / TOTAL_IMAGES
    step = TOTAL_IMAGES/NUM_IMAGES
    transform_original = json.load(open("../data/nerf_synthetic/lego/transforms_test.json"))
    transform_new = copy(transform_original)
    transform_new['frames'] = []
    all_imgs = {i for i in range(TOTAL_IMAGES)}
    train_imgs = {round(step*i) for i in range(NUM_IMAGES)}
    test_imgs = all_imgs - train_imgs
    test_imgs_array = list(test_imgs)
    TEST_IMAGES = 40
    test_step = len(test_imgs) / TEST_IMAGES
    for i in range(len(test_imgs_array)):
        if i % 2 == 0:
            unprocess_image(f"r_{test_imgs_array[i]}")
            transform_new['frames'].append(transform_original['frames'][test_imgs_array[i]])
        with open("../data/nerf_raw/lego/transforms_test.json", "w") as fp:
            json.dump(transform_new, fp, indent=4)
