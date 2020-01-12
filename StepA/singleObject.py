import cv2
import numpy as np
from matplotlib import pyplot as plt

import glob
from PIL import Image
import argparse

import sys

# Function to find object matches using SIFT
def findObjects(model_number, scene_image, model_image):
    sift = cv2.xfeatures2d.SIFT_create()

    # Detecting Keypoints in the two images
    keypoint_query = sift.detect(model_image)
    keypoint_train = sift.detect(scene_image)

    # Computing the descriptors for each keypoint
    keypoint_query, des_query = sift.compute(model_image, keypoint_query)
    keypoint_train, des_train = sift.compute(scene_image, keypoint_train)

    # Matching Algorithm
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Matching the descriptors
    matches = flann.knnMatch(des_query,des_train,k=2)

    # Ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.65 * n.distance:
            good.append(m)

    # Selectinh the top 150 matches
    if len(good)>150:
        
        src_pts = np.float32([keypoint_query[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoint_train[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # find homography matrix and do perspective transform
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = model_image.shape[:2]
        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)
        cv2.imshow(str(model_number), cv2.resize(model_image, (300, 400)))

        # Calculate height, width and centroid on detectd objects
        x = abs(np.int32(dst)[0][0][0] + np.int32(dst)[1][0][0] + np.int32(dst)[2][0][0] + np.int32(dst)[3][0][0])/4
        y = abs(np.int32(dst)[0][0][1] + np.int32(dst)[1][0][1] + np.int32(dst)[2][0][1] + np.int32(dst)[3][0][1])/4

        w = max(np.int32(dst)[0][0][0], np.int32(dst)[1][0][0], np.int32(dst)[2][0][0], np.int32(dst)[3][0][0]) - min(np.int32(dst)[0][0][0], np.int32(dst)[1][0][0], np.int32(dst)[2][0][0], np.int32(dst)[3][0][0])
        h = max(np.int32(dst)[0][0][1], np.int32(dst)[1][0][1], np.int32(dst)[2][0][1], np.int32(dst)[3][0][1]) - min(np.int32(dst)[0][0][1], np.int32(dst)[1][0][1], np.int32(dst)[2][0][1], np.int32(dst)[3][0][1])

        print('Product {} {{position ({},{}), width: {}px, height: {}px}}'.format(model_number, x, y, w, h))

        # Plot found objects
        cv2.polylines(scene_image, [np.int32(dst)], True, (0,255,255), 3, cv2.LINE_AA)

    return scene_image


# Function for reading arguments
def my_args(args=None):
    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument('-models', '--models', help='models directory', required=True)
    parser.add_argument('-scenes', '--scenes', help='scenes directory', required=True)

    results = parser.parse_args(args)
    return results.models, results.scenes


# Load images from directory
def img_load(dir):
    dir += '/*'
    models = glob.glob(dir)
    images = []
    for filename in models:
        images.append(np.array(Image.open(filename)))
    return images


def main():

    all_models, all_scenes = my_args(sys.argv[1:])

    models = img_load(all_models)
    scenes = img_load(all_scenes)

    for i, scene_image in enumerate(scenes):
        print('\nScene {} - '.format(i))

        for j, model_image in enumerate(models):
            scene_image = findObjects(j, scene_image, model_image)

        cv2.imshow('Result ', scene_image)

        RGB_img = cv2.cvtColor(scene_image, cv2.COLOR_BGR2RGB)

        plt.imsave('Result '+ str(i)+ '.jpg', RGB_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()