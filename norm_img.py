####################################################################################################################
###                      *************** Computer Vision / Bioinformatics **************
###                      *** Automatic image normalization and dataset amplification ***
###
### Student: Tiago Filipe dos Santos     Number: 202008971
###
####################################################################################################################

#    All code contained in this script can be used under the term of GNU
#    Affero General Public License.
#
#    Final project for Computer Vision
#
#    Copyright (C) 2021  Tiago F. Santos
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as
#    published by the Free Software Foundation, either version 3 of the
#    License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

#####################################################################################################################
# import libraries
import cv2 as cv
import numpy as np

import pandas as pd
import sklearn as sk

from sklearn.decomposition import PCA

import argparse

def open_img(img):
    """ opens image """
    return cv.imread(img, )

def show_img(img, title = 'Image'):
    """ shows an image within the same window """
    # copy so images are not destroyed by the text
    img_temp = img.copy()

    # add windows title
    cv.putText(img_temp, title, (30, 20),
                cv.FONT_HERSHEY_PLAIN, 1, (255,118,106))

    if debug: print(title, img.shape)

    # show image
    cv.imshow("Image Aligner", img_temp)
    cv.setWindowProperty("Image Aligner", cv.WND_PROP_TOPMOST, 1)
    cv.waitKey(0)

    pass

# create a rotation function
def rotate(img, angle, rot_point = None, debug=False):
    """ rotates an input image """
    # code adapted from https://www.youtube.com/watch?v=oXlwWbU8l2o
    (height, width) = img.shape[:2]
    if rot_point == None :
        rot_point = (width//2, height //2)

    rotMat = cv.getRotationMatrix2D(rot_point, angle, 1)
    dimensions = (width, height)

    img_out = cv.warpAffine(img, rotMat, dimensions)

    if debug: show_img(img_out, 'Rotated Image')

    return img_out

def create_dev_sample(rotation, height = 500, width = 500, debug = True):
    # Base image creation
    height = 500
    width = 500

    # create blank square
    canvas = np.zeros((width, height, 3), dtype='uint8')
    canvas[100:200, 50:250] = 255,255,255
    if debug : show_img(canvas, 'Canvas')

    # misalign the square by theta degrees
    img = rotate(canvas, rotation)

    # save to disk dev sample
    cv.imwrite('image/rect1.jpg', img)

    # show created image
    show_img(img, 'Created Sample Image - Close to continue')

    return img

def img_center(img):
    """ returns the image centroid """
    return (img.shape[0] // 2, img.shape[1] // 2)

def remove_background(img, background, show = True, video = False, debug = True):
    """ background subtraction for static images or videos
        in images, if the pixel has the same color as the blank picture, sutract the picture """
    # TODO for videos
    if video:
        # README https://docs.opencv.org/4.x/d8/d38/tutorial_bgsegm_bg_subtraction.html
        # cv.createBackgroundSubtractorKNN() and cv.createBackgroundSubtractorKNN() are used to infer background from video
        pass
    # for images if pixel if the same, subtract, else = pixel
    img_out = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')

    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if img[i][j] == background[i][j]:
                img_out[i][j] = 0
            else:
                img_out[i][j] = img[i][j]

    # debug
    if debug : show_img(img_out, "Debug")

    return img_out # this is wrong, rework

def assure_gray_img(img):
    """ returns a grayscale image, converting it, if need be """
    img_gray = img if len(img.shape) == 2 else cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    return img_gray

def contrast_stretching(img, debug = True):
    """ does contrast stretching on a grayscale image """

    # assure the image is in grayscale
    img_gray = assure_gray_img(img)

    # initialize image
    img_out = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')

    # calculate extremes
    min, max = np.min(img_gray), np.max(img_gray)
    delta = max - min

    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
             img_out[i][j] = np.int(((img_gray[i][j] - min) / delta)*255)

    if debug : show_img(img_out, "Contrast Streching")

    return img_out

def otsu_thresholding(img, debug = True):
    """ calculates Otsu's threshold after a Gaussian filter

    NOTE: Otsu's thresholding requires grayscale images"""
    # https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html

    blured_img = cv.GaussianBlur(src=img, ksize=(5,5), sigmaX=0)
    value, th_img = cv.threshold(blured_img, thresh = 0, maxval = 255,
                                 type = cv.THRESH_BINARY+cv.THRESH_OTSU)

    if debug: show_img(th_img, "Otsu's thresholding")

    return value

def binarize_image(img, threshold = None, debug = True):
    """ binarizes an image, i.e., conversion to black and white """
    # NOTE: Otsu's thresholding already provides a binarized image, but for the sake of this exercize...

    # assure the image is in grayscale
    img_gray = assure_gray_img(img)

    # contrast stretching
    img_stretch = contrast_stretching(img_gray, debug = debug)

    # set threshold using Otsu's tresholding
    if threshold is None:
        threshold = otsu_thresholding(img_stretch, debug = debug) if threshold is None else threshold
    # NOTE: regular numerical thresholding works better for skull definition (e.g., threshold = 200)

    # initialize BW output
    img_bw = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')

    # for each value above 0, convert to one
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if img_gray[i][j] > threshold:
                img_bw[i][j] = 255

    if debug:
        # test remove_background.
        # Note: background should be a image taken by the sensor when no object is on sight
        # img2 = remove_background(img_gray, img_bw, debug = debug)

        show_img(img_bw, 'Binarized image')

    return img_bw

def denoise_img(img, radius = 13):
    """ denoises an image through median passthrough """
    # cv.imshow('')
    img_out = cv.medianBlur(img, ksize=radius)

    if debug:
        show_img(img_out, 'Denoised image')

    return img_out

def find_boundaries(img, threshold = 125, debug = True):
    """ finds image boundaries horizontal and vertical boundaries """
    min_x, max_x, min_y, max_y = img.shape[1], 0, img.shape[0], 0

    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if img[i][j] > threshold:
                if j < min_x: min_x = j
                if j > max_x: max_x = j
                if i < min_y: min_y = i
                if i > max_y: max_y = i

    boundaries = (min_x, max_x, min_y, max_y)

    if debug : print(boundaries)

    return boundaries

def find_center(img):
    """ find image centroid """
    return img.shape[1] // 2, img.shape[0] // 2

def find_centering_vector(img, boundaries, debug = True):
    """ finds the offset of a segment in relation to the
    center of an image, and returns the centering vector """

    # pass boundaries and vector to a more manageable format
    # print("Boundaries:", boundaries)
    min_x, max_x, min_y, max_y = boundaries

    img_cx, img_cy = find_center(img)
    seg_cx, seg_cy = min_x + (max_x - min_x) // 2, min_y + (max_y - min_y) // 2

    vector = (img_cx - seg_cx, img_cy - seg_cy)

    if debug : print("Vector:", vector)

    return vector

def has_color(img):
    """" check if the image has color or is grayscale """
    color = True if len(img.shape) > 2 else  False

    return color


def move_segment(img, boundaries = None, vector = None, output_shape = None, debug = True):
    """ move one segment of the image to another segment

    Usage:
    move_segment(img, boundaries, vector)

    Where:
    img        - a bidemensional numpy matrix
    boundaries - a list or tuple with the format min_x, max_x, min_y, max_y
    vector     - a list or tuple with the format x, y
    """

    # pass boundaries and vector to a more manageable format
    if boundaries is None:
        min_x, max_x, min_y, max_y = 0, img.shape[1], 0, img.shape[0]
    else:
        min_x, max_x, min_y, max_y = boundaries
        channels = 0
    vector_x, vector_y = vector

    # translate image
    if has_color(img):
        # assure the canvas shape is bigger even when the original when vectors are negative
        canvas = np.zeros((img.shape[0]+np.abs(vector_y),
                           img.shape[1]+np.abs(vector_x),
                           img.shape[2]), dtype='uint8')

        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                for c in range(img.shape[2]):
                    canvas[y + vector_y][x + vector_x][c] = img[y][x][c]

    else:
        canvas = np.zeros((img.shape[0]+np.abs(vector_y),
                           img.shape[1]+np.abs(vector_x)),
                           dtype='uint8')

        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                canvas[y + vector_y][x + vector_x] = img[y][x]

    # crop image to original shape
    img_out = canvas[:img.shape[0], :img.shape[1]]

    # debug
    if debug : show_img(img_out, "Moved segment")

    return img_out

def center_img(img, debug = True):
    """ centers an object in a grayscale image """
    boundaries = find_boundaries(img, debug = False)
    c_vector = find_centering_vector(img, boundaries, debug = False)
    c_img = move_segment(img, boundaries, c_vector, debug = debug)

    return c_img, c_vector

def fill_obj(img, debug = True):
    """ fills object found """
    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    img_filled = cv.fillPoly(img, [contours[0]], color=(255,0,0))

    # debug
    if debug : show_img(img_filled, 'Filled Image')

    return img_filled

def get_proportion(img, threshold = 125, debug = False):
    """ gets the proportion of pixels above a threshold in a rectangular matrix """
    # inialize
    count = 0
    img = assure_gray_img(img)
    total = img.shape[0] * img.shape[1]

    # get proportions
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x][y] > threshold:
                count += 1

    prop = count / total

    if debug == 'Full':
        print("Segment proportion", prop)

    return prop

def get_ratio(template, img, threshold = 10):
    """ returns object proportion between two images """
    prop_template = get_proportion(template, threshold)
    prop_img = get_proportion(img)

    return prop_template / prop_img

def get_segment_poss(img, threshold = None, background = None, debug = True):
    """ returns a list of tuples of each pixel in the segment """
    # based on:
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8540825
        # https://stackoverflow.com/questions/46109338/find-orientation-of-object-using-pca

    ####### preprocessing steps ######
    # TODO create all preprocessing steps from CT images
    #
    # background subtraction (DONE)
    # contrast streching (DONE)
    # Otsu's thresholding (DONE)
    # binarize/thresholding image (DONE)
    # remove noise (DONE)
    # find retangular boundaries (DONE)
    # re-center image (DONE)
    # LCA (HARD) (MAYBE NOT NEEDED)
    # bluring (NO NEED)
    # contour detection - first outer countour - openCV function for this (DONE)
    # holes filling (DONE)
    # get segment positions (DONE)
    # size/area normalization (MEDIUM)

    # convert image to grayscale for easier analysis
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if background is not None:
        img = remove_background(img, background)
        cv.imshow('Object', remove_background(img_gray, img_bw)) # change img_bw in the end of debugging
        cv.waitKey(0)

    # segment image by binarization
    img_bw = binarize_image(img_gray, threshold, debug = debug)

    # denoise image
    img_denoised = denoise_img(img_bw)

    # find image center
    centered_img, c_vector = center_img(img_denoised, debug = debug)

    if centered:
        # FIXME passing centered image makes PCA results wrong even on synthetic example, why??
        # fill the shape of the object
        img_filled = fill_obj(centered_img, debug = debug)
    else:
        img_filled = fill_obj(img_denoised, debug = debug)

    # convert black and white to coordinates for orientation PCA
    obj_poss = []

    # print(img_bw.shape)
    # print(img_filled.shape)

    for i in range(0, img_filled.shape[0]):
        for j in range(0, img_filled.shape[1]):
            if img_filled[i][j] > 0:
                obj_poss.append([i,j])

    # print(obj_poss)

    return obj_poss, c_vector

def find_orientation(obj_pos_matrix, threshold = None, debug = False):
    """ finds the orientation given an image """

    # get segment raster positions
    obj_pos_matrix, c_vector = get_segment_poss(img, threshold = threshold, debug = debug)

    # make PCA on the image to get vectors of major change (PCA's eigenvectors)
    pca = PCA(n_components=2)
    pca.fit(obj_pos_matrix)

    # assign PC1 eigenvector values
    # print(pca.components_)
    x, y = pca.components_[0,0], pca.components_[1,0] # first value of each row is PC1
    # print(pca.components_)

    # debug string
    if debug : print(x,y, -y/x) # for a 45º rotation, x and y should be the same

    # maths reference https://stats.stackexchange.com/questions/239959/how-to-obtain-the-angle-of-rotation-produced-by-a-pca-on-a-2d-dataset
    rot_rad = -np.arctan(y/x)
    rot_deg = np.round(90 - (rot_rad/(np.pi*2))*360, decimals=0)
    if debug : print(f"Rotation angle is:\t {np.int(rot_deg)}º") # sometimes it finds the perpendicular angle

    return rot_deg, c_vector

def find_img_quadrants(img):
    """ returns an ordered list of image quadrants, sorted by their proportions """
    # find image center
    cx, cy = find_center(img)

    # split quadrants by image center
    quadrants = []

    quadrants.append(img[:cy, cx:])    # quad I
    quadrants.append(img[:cy, :cx])    # quad II
    quadrants.append(img[cy:, :cx])    # quad III
    quadrants.append(img[cy:, cx:])    # quad IV

    # find quadrant proportions
    prop_list = []

    for quadrant in quadrants:
        prop_list.append(np.round(get_proportion(quadrant, threshold = 0), decimals = 3))
        # NOTE round proportion to avoid pixelwise errors
        # centered images have all the same proporttions
    if debug : print(prop_list)

    # calculate proportions
    prop_list_order = [1, 2, 3, 4]

    for i in range(0, 3) :
        for j in range(1, 4) :
            if prop_list[i] > prop_list[j]:
                temp = prop_list[i]
                prop_list[i] = prop_list[j]
                prop_list[j] = temp

    if debug: print(prop_list_order)

    return prop_list, prop_list_order


def resize_img(img, ratio = 1, dim_target = None, crop = True, debug = False):
    """ resizes an image to a set ratio """
    # check input format
    # img = assure_gray_img(img)

    # if has_color(img):
    #     dim = (np.int(img.shape[0]*racio), np.int(img.shape[1]*racio), img.shape[2])
    # else:
    dim_img = img.shape[0], img.shape[1]
    dim = (np.int(img.shape[1]*ratio), np.int(img.shape[0]*ratio))

    img_resized = cv.resize(img, dim, interpolation = cv.INTER_CUBIC)
    print(img_resized.shape)

    if crop:
        # resized image centroid
        img_cx, img_cy = find_center(img_resized)


        if img.shape[0] < img_resized.shape[0] or img.shape[1] < img_resized.shape[1]:
            # original image limits
            min_x, max_x, min_y, max_y = (img_cx - (img.shape[1] // 2),
                                        img_cx + (img.shape[1] // 2),
                                        img_cy - (img.shape[0] // 2),
                                        img_cy + (img.shape[0] // 2))
            # create cropped output
            img_out = img_resized[min_x:max_x, min_y:max_y]
        else:             # original image limits
            min_x, max_x, min_y, max_y = ((img.shape[1] // 2) - img_cx,
                                          (img.shape[1] // 2) + img_cx,
                                          (img.shape[0] // 2) - img_cy,
                                          (img.shape[0] // 2) + img_cy)

            # create frame with proper dimensions
            img_out = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype='uint8')

            # fill in with centered resized image
            img_out[min_y:min_y+img_resized.shape[0], min_x:min_x+img_resized.shape[1]] = img_resized
    else:
        img_out = img_resized

    # a target dimension is set, rescale again to target dimension
    if dim_target:
        img_out = cv.resize(img_out, dim_target, interpolation = cv.INTER_CUBIC)


    if debug == 'Full': show_img(img_out, 'Resized image')

    return img_out

def correct_img(img, centering_vector, rotation = 0, resize_ratio = 1):
    """ performs image corrections such as correction and resizing """
    # recenter image
    img_centered = move_segment(img, vector = centering_vector)

    # align image
    img_rotated = rotate(img_centered, rotation, debug = True)

    # make object the same size as reference object
    img_final = resize_img(img_rotated, ratio = resize_ratio,
                           dim_target = (template.shape[1], template.shape[0]),
                           crop = True, debug = False)

    # correct for inversions and PC1 and PC2 reversals
    quadrants_ref   = np.argmin(find_img_quadrants(template))
    quadrants_final = np.argmin(find_img_quadrants(img_final))

    # img_final=rotate(img_final, -90*)


    show_img(img_final, "Transformed result")

    pass

def display_analisys(img, rot_deg):
    """ displays rotation analisys """
    cy, cx = img_center(img) # note - OpenCV uses inverted coordinates
    deltax, deltay = np.int(np.sin(np.deg2rad(rot_deg))*100), np.int(np.cos(np.deg2rad(rot_deg))*100)

    cv.putText(img, "Image has a "+str(rot_deg)+" or "+str(rot_deg+180)+" degrees tilt", (img.shape[1] // 2 - 110, 50),
                cv.FONT_HERSHEY_PLAIN, 1, (255,118,106))
    cv.line(img, (cx + deltax, cy + deltay), (cx - deltax, cy - deltay), (125,125,0))

    show_img(img, title='Result')

def parse_args():
    parser = argparse.ArgumentParser(
        "automatic image allignment and library with preprocessing tools ")
    parser.add_argument("-i", "--infile", required=False, help="input image file name",
                        default="image/brain1.jpg")
    parser.add_argument("-d", "--debug", required=False, help="set debug mode",
                        default=False)
    parser.add_argument("-r", "--reference", required=False, help="set reference image for corrections",
                        default=None)


    return parser.parse_args()

if __name__ == "__main__":
    """ automatic image alignment and library with preprocessing tools """
    # parse arguments
    args = parse_args()

    # set debug switch here
    debug = args.debug
    debug = True

    # define parameters
    rotation = 80 # FIXME on synthetic example rotation=]0-2, 20, 160-170]º returns rot_deg=180-theta, why??
                  # FIXME on real example is misaligned by two degrees, and it is often is 180 - theta

    # set run type
    use_synthetic_example = True

    # load image and process it
    if use_synthetic_example:
        img = create_dev_sample(rotation)
        centered = False # alignment with decent results
    else:
        img = rotate(open_img(args.infile), rotation)  # TODO not great alignment results, Why?
        centered = True

    # define the template image to be used for alignments
    try:
        if template :
            template = reference
            if debug: print('Using user provided template')
    except:
        template = resize_img(open_img(args.infile), 0.4, crop=True)

    # show template and image to be aligned
    cv.imshow("Reference Image", template)
    cv.waitKey(0)
    cv.imshow("Original Image", img)
    cv.waitKey(0)

    # copy image to avoid OpenCV destructive processing
    orig_img = img.copy()

    # calculate orientation and offset
    rot_deg, c_vector = find_orientation(img, debug = debug)

    # display rotation and result
    if debug : display_analisys(img, rot_deg)

    # get template/image resize ratio
    ratio = get_ratio(template, img)

    # display corrected image
    correct_img(orig_img, c_vector, -rot_deg, ratio)

    # close all OpenCV windows
    cv.destroyAllWindows()

    ############## tests #############

    # confirm PCA provided angles
    assert np.abs(rotation - rot_deg) < 3 or np.abs(180 - rotation - rot_deg) < 3 ,\
        print(f"\n ERROR: PCA detected rotation, {rot_deg} is different from provided rotation {rotation} or {180+rotation}")
        # algo cannot yet recognize direction


    # TODO feed to ImageNet and classify
    # TODO compare classification of aligned images to original images classification
    # TODO catalog z-axis orientation through classification

#######################################################################################################
###                                          Referrences                                            ###
#######################################################################################################

# Automatic Image Alignment Using Principal Component Analysis
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8540825
# OpenCV Course - Full Tutorial with Python: https://www.youtube.com/watch?v=oXlwWbU8l2o
# OpenCV Documentation - https://docs.opencv.org/
# StackExchange - various issues and doubts
    #   https://stats.stackexchange.com/questions/239959/how-to-obtain-the-angle-of-rotation-produced-by-a-pca-on-a-2d-dataset
# StackOverflow - various issues and doubts
    #   https://stackoverflow.com/questions/46109338/find-orientation-of-object-using-pca
# TowardsDataScience
    # https://towardsdatascience.com/a-step-by-step-implementation-of-principal-component-analysis-5520cc6cd598
# SciKit Online Manual https://scikit-learn.org