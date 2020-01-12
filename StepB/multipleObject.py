import glob
from PIL import Image
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt

import sys

def main():

    all_models, all_scenes = my_args(sys.argv[1:])

    models = img_load(all_models)
    scenes = img_load(all_scenes)

    for i, scene_image in enumerate(scenes):
        print('\nScene {} - '.format(i))

        scene_image_plot = scene_image.copy()

        for j, model_image in enumerate(models):
            occur = 0
            data = []
            scene_image1, occur, data, scene_image_plot1 = findObjects(j, scene_image, model_image, occur, data, scene_image_plot)
            scene_image2, occur, data, scene_image_plot2 = findObjects(j, scene_image1, model_image, occur, data, scene_image_plot1)

            if(occur==1):
                print('Product {} - {} instance found:'.format(j,occur))
                print('Instance {} {{position: ({},{}), width: {}px, height: {}px}}'.format(occur, data[0][1], data[0][2], data[0][3], data[0][4]))

            if(occur==2):
                print('Product {} - {} instance found:'.format(j,occur))
                print('Instance {} {{position: ({},{}), width: {}px, height: {}px}}'.format(occur-1, data[0][1], data[0][2], data[0][3], data[0][4]))
                print('Instance {} {{position: ({},{}), width: {}px, height: {}px}}'.format(occur, data[1][1], data[1][2], data[1][3], data[1][4]))

        cv2.imshow('Result', scene_image_plot2)

        RGB_img = cv2.cvtColor(scene_image_plot2, cv2.COLOR_BGR2RGB)

        plt.imsave('Result '+ str(i)+ '.jpg', RGB_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Function to find object matches using SIFT
def findObjects(model_number, scene_image_param, model_image, count, data, scene_image_plot_param):
    sift = cv2.xfeatures2d.SIFT_create()

    # Detecting Keypoints in the two images
    keypoint_query = sift.detect(model_image)
    keypoint_train = sift.detect(scene_image_param)

    # Applying template matching to the images
    '''
    img_gray = cv2.cvtColor(scene_image, cv2.COLOR_BGR2GRAY)
    model_gray = cv2.cvtColor(model_image, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray,model_image,cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        im2 = cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
    '''

    # Computing the descriptors for each keypoint
    keypoint_query, des_query = sift.compute(model_image, keypoint_query)
    keypoint_train, des_train = sift.compute(scene_image_param, keypoint_train)

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
        if m.distance < 0.8 * n.distance:
            good.append(m)

    scene_image_r = scene_image_param
    scene_image_plot_r = scene_image_plot_param

    # Selectinh the top 150 matches
    if len(good)>150:
        
        src_pts = np.float32([keypoint_query[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoint_train[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # find homography matrix and do perspective transform
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = model_image.shape[:2]
        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)

        # Calculate height, width and centroid on detectd objects
        x = abs(np.int32(dst)[0][0][0] + np.int32(dst)[1][0][0] + np.int32(dst)[2][0][0] + np.int32(dst)[3][0][0])/4
        y = abs(np.int32(dst)[0][0][1] + np.int32(dst)[1][0][1] + np.int32(dst)[2][0][1] + np.int32(dst)[3][0][1])/4

        w = max(np.int32(dst)[0][0][0], np.int32(dst)[1][0][0], np.int32(dst)[2][0][0], np.int32(dst)[3][0][0]) - min(np.int32(dst)[0][0][0], np.int32(dst)[1][0][0], np.int32(dst)[2][0][0], np.int32(dst)[3][0][0])
        h = max(np.int32(dst)[0][0][1], np.int32(dst)[1][0][1], np.int32(dst)[2][0][1], np.int32(dst)[3][0][1]) - min(np.int32(dst)[0][0][1], np.int32(dst)[1][0][1], np.int32(dst)[2][0][1], np.int32(dst)[3][0][1])

        # Extract data if match found and pasing it to the main function
        if(w>300 and h > 300):
            current_data = [model_number, x, y, w, h]
            data.append(current_data)
            count += 1

            # Plot the outline of the product on a copy of the original image
            cv2.polylines(scene_image_plot_r, [np.int32(dst)], True, (0,255,0), 3, cv2.LINE_AA)

            # Mask the detected product with a filled polygon
            cv2.fillPoly(scene_image_r, [np.int32(dst)], 255)
            
    return scene_image_r, count, data, scene_image_plot_r


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

if __name__ == "__main__":
    main()