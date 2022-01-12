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

from scipy.spatial import distance
from scipy import fftpack as fft
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn as sk
from sklearn.decomposition import PCA

import argparse

from cv_preprocess_lib import *

def create_dev_sample(rotation, shape = 'rectangle', height = 500, width = 500, debug = True):
    # Base image creation
    height = 500
    width = 500

    # create blank square
    canvas = np.zeros((width, height, 3), dtype='uint8')
    if shape == 'triangle':
        # draw a triangle
        vertices = np.array([[100, 100], [250, 400], [400, 100]], np.int32)
        pts = vertices.reshape((-1, 1, 2))
        cv.polylines(canvas, [pts], isClosed=True, color=(255, 255, 255), thickness=20)

        # fill it
        cv.fillPoly(canvas, [pts], color=(255, 255, 255))
    else:
        canvas[100:200, 50:250] = 255,255,255

    if debug : show_img(canvas, 'Canvas')

    # misalign the square by theta degrees
    img = rotate(canvas, rotation)

    # save to disk dev sample
    cv.imwrite('images /rect1.jpg', img)

    # show created image
    show_img(img, 'Created Sample Image - Close to continue')

    return img

def find_n_axis(img, contour = None, center = None):
    """ returns the number of axis of simmetry of the image """
    # NOTE check also http://www.cse.psu.edu/~yul11/CourseFall2006_files/loy_eccv2006.pdf
    #      code in https://github.com/dramenti/symmetry

    # initialize
    if contour is None :
        contour = find_best_contour(img, debug = False)
        # if debug :
        #     print("No contour passed")
        #     print(contour)
    # else:
    #     print("Contour passed\n", contour)

    # create image
    # centered_img = preprocess_img(img, debug = False)
    # contour_img = cv.drawContours(cv.cvtColor(centered_img ,cv.COLOR_GRAY2RGB), contour, -1, (0,255,0), 3)
    contour_img = cv.drawContours(img, contour, -1, (0,255,0), 3)

    # set center
    if center :
        print(center)
        cx, cy = center
    else:
        cx, cy = center = find_center(img)

    # calculate list with distances to each external countour point
    dist_list = []
    angle_list = []
    for edge in contour:
        # print(center, edge)
        dist_list.append(distance.euclidean(center, *edge))
        angle_list.append(calculate_angle(center, *edge))
        lists_merged = zip(center, edge)
        # print(lists_merged)
    sort_angles = sorted(zip(angle_list, dist_list), key=lambda x: x[0])

    # sort list by angle
    dist_list = []
    angle_list = []
    for angle, dist in sort_angles:
        dist_list.append(dist)
        angle_list.append(angle)
        lists_merged = zip(center, edge)

    # calculate mean angle step
    angle_1 = angle_list[1:]
    angle_2 = angle_list[:-1]
    angle_diffs = [angle_1 - angle_2 for angle_1, angle_2 in zip(angle_1, angle_2)]
    mean_angle = np.mean(angle_diffs)

    # Fourier transform the distance versus angle function to get the period of the contour
    # based on https://scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_periodicity_finder.html

    # calculate Fourier transform and FT frequencies
    ft = fft.fft(dist_list, axis = 0)
    ft_freqs = fft.fftfreq(n = len(dist_list), d = mean_angle)

    # fix for div by 0 in ft_periods
    ft_freqs_2 = np.where(0, 1e-20, ft_freqs)

    # calculate function period
    ft_periods = 1 / ft_freqs_2

    # crop negative angles and first value since it has border condition bias in their calculations
    # print(ft_periods)
    ft_cropped = ft[1:len(ft)//2]
    ft_periods_cropped = ft_periods[1:len(ft)//2]

    # get the angular period
    prob_period = ft_periods_cropped[np.argmax(np.abs(ft_cropped))]
    print("Angular Period", prob_period)

    # get number of simmetry angles
    n_axis = np.int(360/prob_period)
    n_axis = n_axis // 2 if n_axis % 2 == 0 else n_axis

    print("Probable number of axis:", n_axis)

    if debug:
        print("Mean angle step:", mean_angle)
        #     print(dist_list)
        #     print(angle_list)
        # print("Sorted_angles list\n\n" , sort_angles)

        # debug period
        # print("length Periods", len(ft_periods), "Max Periods", max(ft_periods), ft_periods[np.argmax(ft)])

        # pretify plots
        sns.set_theme()
        plt.rcParams["figure.figsize"] = (12, 6.75)
        plt.rcParams["axes.titlesize"] = "large"
        plt.rcParams["axes.labelsize"] = "medium"
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.sans-serif'] = ['Times New Roman']

        # get most relevant period for FFT frequencies list
        fig = plt.figure()
        fig.set_tight_layout(tight=True)

        # set subplots
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        # countour distance plot
        s_contour = np.vstack((x.reshape(-1,2) for x in contour)) # https://stackoverflow.com/questions/52206407/simplify-getting-coordinates-from-cv2-findcontours
        # print(s_contour)
        ax1.plot(angle_list, dist_list, marker='o')
        ax1.set(xlabel = 'Point in Countour List', ylabel = 'Euclidean Distance',
                title = 'Countour Distance to Center Function')

        # get most relevant period for FFT frequencies list
        ax2.plot(ft_periods, abs(ft), marker='o')
        ax2.set(xlim = (0, 360),
                xlabel = 'Period (degrees)', ylabel = 'Power',
                title = 'Relative Importance of Each Fourier Period')
        plt.show()

    cv.putText(contour_img, "  Image has a period of aprox. "+str(np.round(prob_period))+", which suggests "+ str(n_axis)+" axis.", (50, 50), cv.FONT_HERSHEY_PLAIN, 1, (255,118,106))

    return n_axis, contour_img

def find_orientation(img, threshold = None, preprocessed = False, c_vector = None, debug = False):
    """ finds the orientation given an image """

    # get segment raster positions
    obj_pos_matrix, c_vector = get_segment_poss(img, threshold = threshold, preprocessed = preprocessed, c_vector = c_vector, debug = debug)

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


def display_analysis(img, rot_deg, output = False, orig_img = None, app = False):
    """ displays rotation analisys """
    # avoid destructive edits with putText() and line()
    img_orig = img.copy()

    cy, cx = img_center(img) # note - OpenCV uses inverted coordinates

    deltax, deltay = np.int(np.sin(np.deg2rad(rot_deg))*100), np.int(np.cos(np.deg2rad(rot_deg))*100)

    cv.putText(img, "Image has a "+str(rot_deg)+" or "+str(rot_deg+180)+" degrees tilt", (img.shape[1] // 2 - 110, 50),
                cv.FONT_HERSHEY_PLAIN, 1, (255,118,106))
    cv.line(img, (cx + deltax, cy + deltay), (cx - deltax, cy - deltay), (125,125,0))

    if not app : show_img(img, title='Result')

    if output:
        return img
    else:
        img = img_orig

def correct_img(img, centering_vector, rotation = 0, threshold = 20, resize_ratio = 1, template = None, debug = True, app = False):
    """ performs image corrections such as correction and resizing """

    if app : debug = False

    # contrast stretching
    img_contrast = contrast_stretching(img, debug = debug)

    # recenter image
    img_centered = move_segment(img_contrast, vector = centering_vector, debug = debug)

    # align image
    img_rotated = rotate(img_centered, rotation, debug = debug)

    # conform to template
    try:
        # contrast stretch template
        template_cs = contrast_stretching(template, debug = debug)

        # get need for orientation correction
        needs_rot = needs_rotation(template_cs, img_rotated)

        if needs_rotation != 0 :
            img_rotated = rotate(img_rotated, 90 * needs_rot, debug = debug)
            if debug : print("Image rotated", 90 * needs_rot, "degrees")

        # make object the same size as reference object
        resize_ratio = get_ratio(template_cs, img_rotated, threshold = 20, debug = debug)
        if debug :
            print("Image Ratio", resize_ratio)

        img_final = resize_img(img_rotated, ratio = resize_ratio,
                            dim_target = (template.shape[1], template.shape[0]),
                            crop = True, debug = debug)
    except:
        print("Template not used.")
        img_final = img_rotated.copy()
        needs_rot = 0

    if not app :
        show_img(img_final, "Transformed result")

    return img_final, needs_rot * 90

def parse_args():
    parser = argparse.ArgumentParser(
        "automatic image allignment and library with preprocessing tools ")
    parser.add_argument("-i", "--infile", required=False, help="input image file name",
                        default="images/brain2.png")
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
    use_synthetic_example = False

    # load image and process it
    if use_synthetic_example:
        img = create_dev_sample(rotation)
        # img = create_dev_sample(rotation, shape = 'triangle')
        centered = False # alignment with decent results
    else:
        img = rotate(open_img(args.infile), rotation)
        centered = True

    # define the template image to be used for alignments
    if args.reference != None :
        template = open_img(args.reference)
        if debug: print('Using user provided template')
    else:
        template = resize_img(open_img(args.infile), 0.4, crop=True)

    # show template and image to be aligned
    cv.imshow("Reference Image", template)
    cv.waitKey(0)
    cv.imshow("Original Image", img)
    cv.waitKey(0)

    # copy image to avoid OpenCV destructive processing
    orig_img = img.copy()

    # find best contour
    contour = find_best_contour(img)

    # calculate axis
    find_n_axis(img, contour)

    # calculate orientation and offset
    rot_deg, c_vector = find_orientation(orig_img, debug = debug)


    # display rotation and result
    if debug :
        display_analysis(img, rot_deg)

    # display corrected image
    img_final, needs_rot = correct_img(orig_img, c_vector, -rot_deg, template = template)


    # close all OpenCV windows
    cv.destroyAllWindows()

    ############## tests #############

    # confirm PCA provided angles
    assert np.abs(rotation - rot_deg + needs_rot) < 3 or np.abs(180 - rotation - rot_deg + needs_rot) < 3 ,\
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