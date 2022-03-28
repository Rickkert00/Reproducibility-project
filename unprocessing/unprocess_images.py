import unprocess
import tensorflow as tf
from plantcv import plantcv as pcv

if __name__ == '__main__':
    # Read RGB image
    img = tf.io.read_file("data/r_0.png")
    png_file = tf.io.decode_png(img, channels=3)
    white_level = 255.0
    png_image = tf.cast(png_file, tf.float32) / white_level
    # RGBimage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # Output img with window name as 'image'
    # cv2.imshow('image', img)
    raw_img, metadata= unprocess.unprocess(png_image)
    tf.keras.utils.save_img(
        'test.tiff', raw_img.numpy(), scale=False)
    print(raw_img)
    pcv.params.debug = "print"

    # read in image
    img, path, img_filename = pcv.readbayer("test.tiff")
    # cv2.imshow('image2', raw_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()