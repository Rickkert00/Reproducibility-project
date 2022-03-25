import cv2
import unprocess

if __name__ == '__main__':
    # Read RGB image
    img = cv2.imread('../data/nerf_synthetic/lego/train/r_0.png')
    RGBimage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Output img with window name as 'image'
    cv2.imshow('image', img)
    raw_img = unprocess.unprocess(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()