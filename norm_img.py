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
    cv.imwrite('images/rect1.jpg', img)

    # show created image
    show_img(img, 'Created Sample Image - Close to continue')

    return img


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

def correct_img(img, centering_vector, rotation = 0, threshold = 20, resize_ratio = 1,
                template = None, debug = True, app = False,
                checklist = ['Contrast', 'Center', 'Align', 'Orientate', 'Size']):
    """ performs image corrections such as correction and resizing """

    if app : debug = False

    # contrast stretching
    if "Contrast" in checklist:
        img_contrast = contrast_stretching(img, debug = debug)
    else:
        img_contrast = img.copy()

    # recenter image
    if "Center" in checklist:
        img_centered = move_segment(img_contrast, vector = centering_vector, debug = debug)
    else:
        img_centered = img_contrast.copy()

    # align image
    if "Align" in checklist:
        img_rotated = rotate(img_centered, rotation, debug = debug)
    else:
        img_rotated = img_centered.copy()


    # conform to template
    try:
        # contrast stretch template
        template_cs = contrast_stretching(template, debug = False)

        if "Orientate" in checklist:
            # main axis correction
            needs_rot = needs_rotation(template_cs, img_rotated)

            if needs_rotation != 0 :
                img_rotated = rotate(img_rotated, 90 * needs_rot, debug = debug)
                if debug : print("Image rotated", 90 * needs_rot, "degrees")

            # image orientation correction
            img_orientated = orientate(img_rotated, template_cs, debug = debug)
            # FIXME fails because template also needs recentering

        else:
            img_orientated = img_rotated.copy()
            needs_rot = 0

        # make object the same size as reference object

        if "Size" in checklist:
            resize_ratio = get_ratio(template_cs, img_orientated, threshold = 20, debug = debug)
            if debug :
                print("Image Ratio", resize_ratio)

            img_final = resize_img(img_orientated, ratio = resize_ratio,
                                dim_target = (template.shape[1], template.shape[0]),
                                crop = True, debug = debug)
        else:
            img_final = img_orientated.copy()

    except Exception as e:
        print("Template not used.")

        if debug:
            print(e)

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
    rotation = 80

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
    _, contour_analysis = find_n_axis(img.copy(), contour, debug = True)
    show_img(contour_analysis, "Contour Analysis")

    # calculate orientation and offset
    rot_deg, c_vector = find_orientation(orig_img, debug = debug)

    # display rotation and result
    if debug :
        display_analysis(img, rot_deg)

    # if "Fill" in checklist:

    # display corrected image
    img_final, needs_rot = correct_img(orig_img, c_vector, -rot_deg, template = template)


    # close all OpenCV windows
    cv.destroyAllWindows()

    ############## tests ############# FIXME Up to date with recent orientation function addition

    # confirm PCA provided angles
    # assert np.abs(rotation - rot_deg + needs_rot) < 3 or np.abs(180 - rotation - rot_deg + needs_rot) < 3 ,\
    #     print(f"\n ERROR: PCA detected rotation, {rot_deg} is different from provided rotation {rotation} or {180+rotation}")


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