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

from norm_img_v4 import *

import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

from dash import Input, Output, dcc, html

import flask
import glob
import os
import webbrowser

import plotly.graph_objects as go
import plotly.express as px

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP],
     suppress_callback_exceptions=True)   #initialising dash app

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "25rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "10rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

# define sample list
image_directory = 'images/'
list_of_images = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory))]
static_image_route = '/images/'

# global variables
threshold = 125
c_vector = (0, 0) # store globally c_vector for use in later functions
rot_deg = 0       # store globally template orientation for use in later functions
obj_poss = []     # used to convert black and white to coordinates for orientation PCA

# configure sidebar
sidebar = html.Div(
    [
        html.H2("Preprocessing", className="display-4"),
        html.Hr(),
        html.Button('Process next', id='button-val', n_clicks=0),
        html.P(),
        html.P("Template to load:", className="lead"),
        dcc.Dropdown(
            id='template-dropdown',
            options=[{'label': i, 'value': i} for i in list_of_images],
            value='brain1.png',
            style={'width': '25vh'}
        ),
        html.P(),
        html.P("Sample to load:", className="lead"),
        dcc.Dropdown(
            id='image-dropdown',
            options=[{'label': i, 'value': i} for i in list_of_images],
            value='brain2.png',
            style={'width': '25vh'}
        ),
        html.P(),
        html.P("Rotation to apply", className="H6"),
        dcc.Slider(id='rotation-slider', min=0, max=360, step=1, value=80,
        ),
        html.P(),
        html.P("Resize to apply", className="H6"),
        dcc.Slider(id='resize-slider', min=0, max=2, step=0.1, value=0.3,
        ),
        html.P(),
        html.P("Preprocessing steps", className="lead"),
        html.P("Segmentation threshold", className="H6"),
        dcc.Slider(id='binarizer-slider', min=0, max=255, step=1, value=125,
        ),
        dbc.Nav(
            [
                dcc.Checklist(
                    id = 'cl-preproc',
                    options=[
                        {'label': 'Remove background', 'value': 'RmBG'},
                        {'label': 'Contrast Streching', 'value': 'Contrast'},
                        {'label': 'Otsu\'s Thresholding', 'value': 'Otsu'},
                        {'label': 'Simple Thresholding', 'value': 'Thresholding'},
                        {'label': 'Denoise Image', 'value': 'Denoise'},
                        {'label': 'Recenter Image', 'value': 'Center'},
                        {'label': 'Fill Object', 'value': 'Fill'},
                    ],
                    value=['RmBG', 'Contrast', 'Otsu', 'Denoise', 'Center', 'Fill'],
                    style={'display':'inline-block', 'width':'45%', 'border':'2px grey solid'}
                )
            ],
            vertical=True,
            pills=True,
        ),
        html.P(),
        html.P("Processing steps", className="lead"),
        dbc.Nav(
            [
                dcc.Checklist(
                    id = 'cl-process',
                    options=[
                        {'label': 'Match Size', 'value': 'Size'},
                        {'label': 'Image Alignment', 'value': 'Align'},
                        {'label': 'Orientation Adjustment', 'value': 'Orientate'},
                    ],
                    value=['Size', 'Center', 'Align', 'Orientate'],
                    style={'display':'inline-block', 'width':'45%', 'border':'2px grey solid'}
                )
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

# configure content area
content =  html.Div(id = 'parent', children = [
    html.H1(id = 'H1', children = 'Image Normalizer', style = {'textAlign':'center',\
                                            "background-color": "#f8f9fa",
                                            'marginLeft': 200, 'marginTop': 0, 'marginBottom':0}),
    html.P("This shows the pre-processing pipeline until the image is normalized. ",
                className="lead", style = { 'textAlign':'left', "background-color": "#f8f9fa",
                                            'marginLeft': 400, 'marginTop': 0, 'marginBottom':0}),
    html.P("The main purpose is to reduce variance before augmentation for machine learning training, by correcting size, rotation and orientation.",
                className="lead", style = { 'textAlign':'left', "background-color": "#f8f9fa",
                                            'marginLeft': 400, 'marginTop': 0, 'marginBottom':0}),
    html.P("Several stats are calculated for preprocessing and shown when adequate." ,
                className="lead", style = { 'textAlign':'left', "background-color": "#f8f9fa",
                                            'marginLeft': 400, 'marginTop': 0, 'marginBottom':40}),

        # html.Img(id='image', style={'marginLeft':400, 'width': '300px', 'horizontal-align': 'center'}),
        html.Img(id='template-image', style={'marginLeft':400, 'width': '300px', 'horizontal-align': 'center'}),
        html.Img(id='image1', style={'marginLeft':15, 'width': '300px', 'horizontal-align': 'center'}),
        html.Img(id='image2', style={'marginLeft':15, 'width': '300px', 'horizontal-align': 'center'}),
        html.Img(id='image3', style={'marginLeft':15, 'width': '300px', 'horizontal-align': 'center'}),
        html.Img(id='image4', style={'marginLeft':400, 'width': '300px', 'horizontal-align': 'center'}),
        html.Img(id='image5', style={'marginLeft':15, 'width': '300px', 'horizontal-align': 'center'}),
        html.Img(id='image6', style={'marginLeft':15, 'width': '300px', 'horizontal-align': 'center'}),
        html.Img(id='image7', style={'marginLeft':15, 'width': '300px', 'horizontal-align': 'center'}),
        html.Img(id='image8', style={'marginLeft':400, 'width': '300px', 'horizontal-align': 'center'}),
        html.Img(id='image9', style={'marginLeft':15, 'width': '300px', 'horizontal-align': 'center'}),
        ]
                     )

# template updater
@app.callback(
    dash.dependencies.Output('template-image', 'src'),
    [dash.dependencies.Input('template-dropdown', 'value'),
     ])
def update_image_src_0(value):
    return static_image_route + value


# image updater
@app.callback(
    dash.dependencies.Output('image', 'src'),
    [dash.dependencies.Input('image-dropdown', 'value'),
     dash.dependencies.Input('rotation-slider', 'value'),
     dash.dependencies.Input('resize-slider', 'value'),
     ])
def update_image_src_0(value, rotation, ratio):
    return static_image_route + value

# image 1 updater - Initial image rotation
@app.callback(
    dash.dependencies.Output('image1', 'src'),
    [dash.dependencies.Input('image-dropdown', 'value'),
     dash.dependencies.Input('rotation-slider', 'value'),
     dash.dependencies.Input('resize-slider', 'value'),
     dash.dependencies.Input('button-val', 'n_clicks'),
    ])
def update_image_src_1(value, rotation, ratio, n_clicks):
    global orig_img
    if n_clicks == 1:
        orig_img = open_img(image_directory + value)
        img_rot = rotate(open_img(image_directory + value), rotation)
        img = resize_img(img_rot, ratio = ratio)
        cv.imwrite('images/image1.png', img)
        return '/images/image1.png'
    elif n_clicks > 1:
        return '/images/image1.png'

# image 2 updater - Contrast Stretching
@app.callback(
    dash.dependencies.Output('image2', 'src'),
    [dash.dependencies.Input('cl-preproc', 'value'),
     dash.dependencies.Input('image1', 'children'),
     dash.dependencies.Input('binarizer-slider', 'value'),
    dash.dependencies.Input('button-val', 'n_clicks'),])
def update_image_src_2(checklist, img, threshold_int, n_clicks):
    global threshold
    if n_clicks == 2 :
        threshold = threshold_int
        img_gray = assure_gray_img(open_img(image_directory + 'image1.png'))

        if "Contrast" in checklist:
            img_stretch = contrast_stretching(img_gray, debug = debug)
            cv.imwrite('images/img_contrast.png', img_stretch)
        else:
            cv.imwrite('images/img_contrast.png', img_gray)

        return '/images/img_contrast.png'
    elif n_clicks > 2:
        return '/images/img_contrast.png'

# image 2 updater - Thresholding
@app.callback(
    dash.dependencies.Output('image3', 'src'),
    [dash.dependencies.Input('cl-preproc', 'value'),
     dash.dependencies.Input('image2', 'children'),
     dash.dependencies.Input('binarizer-slider', 'value'),
     dash.dependencies.Input('button-val', 'n_clicks'),])
def update_image_src_3(checklist, img, threshold_int, n_clicks):
    global threshold
    if n_clicks == 3:
        threshold = threshold_int
        img_gray = assure_gray_img(open_img(image_directory + 'img_contrast.png'))

        if "Thresholding" in checklist:
            img_bw = binarize_image(img_gray, threshold, debug = debug)
            cv.imwrite('images/image2.png', img_bw)
        elif "Otsu" in checklist:
            img_bw = otsu_thresholding(img_gray, debug = False, app = True)
            cv.imwrite('images/image2.png', img_bw)
        else:
            cv.imwrite('images/image2.png', img_gray)

        return '/images/image2.png'
    elif n_clicks > 3:
        return '/images/image2.png'

# image 3 updater - Denoiser
@app.callback(
    dash.dependencies.Output('image4', 'src'),
    [dash.dependencies.Input('cl-preproc', 'value'),
     dash.dependencies.Input('image3', 'children'),
     dash.dependencies.Input('binarizer-slider', 'value'),
     dash.dependencies.Input('button-val', 'n_clicks'),])
def update_image_src_4(checklist, img, threshold, n_clicks):
    if n_clicks == 4:
        img_bw = assure_gray_img(open_img(image_directory + 'image2.png'))
        print("####Checklist###", checklist)
        if "Denoise" in checklist:
            img_denoised = denoise_img(img_bw, debug = False)
            cv.imwrite('images/image3.png', img_denoised)
        else:
            cv.imwrite('images/image3.png', img_bw)

        return '/images/image3.png'
    elif n_clicks > 4:
        return '/images/image3.png'


# image 4 updater - Center Image
@app.callback(
    dash.dependencies.Output('image5', 'src'),
    [dash.dependencies.Input('cl-preproc', 'value'),
     dash.dependencies.Input('image4', 'children'),
     dash.dependencies.Input('binarizer-slider', 'value'),
     dash.dependencies.Input('button-val', 'n_clicks'),])
def update_image_src_5(checklist, img, threshold, n_clicks):
    if n_clicks == 5:
        global c_vector
        img_denoised = assure_gray_img(open_img(image_directory + 'image3.png'))

        if "Center" in checklist:
            centered_img, c_vector = center_img(img_denoised, debug = False)
            cv.imwrite('images/image4.png', centered_img)
        else:
            cv.imwrite('images/image4.png', img_denoised)
        return '/images/image4.png'
    elif n_clicks > 5:
        return '/images/image4.png'

# image contour updater - Contours
@app.callback(
    dash.dependencies.Output('image6', 'src'),
    [dash.dependencies.Input('image5', 'children'),
     dash.dependencies.Input('button-val', 'n_clicks'),])
def update_image_src_6(img, n_clicks):
    if n_clicks == 6:
        # show contour used in the previous image
        img_filled = assure_gray_img(open_img(image_directory + 'image4.png'))
        contour =   find_best_contour(img_filled, preprocessed = True, debug = False, app = True)
        _, contour = find_n_axis(img_filled)
        # show_img(contour, "Countour Drawn"))
        cv.imwrite('images/contour.png', contour)

        return '/images/contour.png'
    elif n_clicks > 6:
        return '/images/contour.png'

# image 5 updater - Fill in the gaps
@app.callback(
    dash.dependencies.Output('image7', 'src'),
    [dash.dependencies.Input('cl-preproc', 'value'),
     dash.dependencies.Input('image6', 'children'),
     dash.dependencies.Input('binarizer-slider', 'value'),
     dash.dependencies.Input('button-val', 'n_clicks'),])
def update_image_src_7(checklist, img , threshold, n_clicks):
    if n_clicks == 7:
        centered_img = assure_gray_img(open_img(image_directory + 'image4.png'))
        if "Fill" in checklist:
            img_filled = fill_obj(centered_img, debug = False)
            cv.imwrite('images/image5.png', centered_img)
        else:
            img_filled = centered_img
            cv.imwrite('images/image5.png', centered_img)

        # get object position for further analysis
        global obj_poss
        for i in range(0, img_filled.shape[0]):
            for j in range(0, img_filled.shape[1]):
                if img_filled[i][j] > 0:
                    obj_poss.append([i,j])

        return '/images/image5.png'
    elif n_clicks > 7:
        return '/images/image5.png'

# image 6 updater - Image Analysis
@app.callback(
    dash.dependencies.Output('image8', 'src'),
    [dash.dependencies.Input('image7', 'children'),
     dash.dependencies.Input('binarizer-slider', 'value'),
     dash.dependencies.Input('button-val', 'n_clicks'),])
def update_image_src_8(img, threshold, n_clicks):
    if n_clicks == 8:
        filled_img = open_img(image_directory + 'image5.png')
        global c_vector
        global rot_deg
        rot_deg, c_vector = find_orientation(img = filled_img, preprocessed = True, c_vector = c_vector, debug = debug)
        analysis = display_analysis(filled_img, rot_deg, output = True, orig_img = 'images/image1.png', app = True)
        cv.imwrite('images/image6.png', analysis)

        return '/images/image6.png'
    elif n_clicks > 8:
        return '/images/image6.png'

# image 7 updater - Correct Image
@app.callback(
    dash.dependencies.Output('image9', 'src'),
    [dash.dependencies.Input('image1', 'children'),
     dash.dependencies.Input('image8', 'children'), # only for updating
     dash.dependencies.Input('template-dropdown', 'value'),
     dash.dependencies.Input('binarizer-slider', 'value'),
     dash.dependencies.Input('button-val', 'n_clicks'),])
def update_image_src_9(img, _, template, threshold, n_clicks):
    if n_clicks == 9:
        global c_vector
        global rot_deg

        orig_img = open_img('images/image1.png')
        template = open_img('images/'+ template)
        # display corrected image
        img_final, needs_rot = correct_img(orig_img, c_vector, -rot_deg, template = template, debug = True, app = False)
        cv.imwrite('images/image7.png', img_final)

        return '/images/image7.png'
    elif n_clicks > 9:
        return '/images/image7.png'

# reset clicks on image change
@app.callback(
    dash.dependencies.Output('button-val', 'n_clicks'),
    [dash.dependencies.Input('image-dropdown', 'value'),
     dash.dependencies.Input('rotation-slider', 'value'),
     dash.dependencies.Input('resize-slider', 'value'),
    # dash.dependencies.Input('button-val', 'n_clicks'),
    ])
def click_reseter_1(value, rotation, ratio):
    global orig_img
    orig_img = open_img(image_directory + value)
    return 0

# Add a static image route that serves images from desktop
@app.server.route('{}<image_path>.png'.format(static_image_route))
def serve_image(image_path):
    image_name = '{}.png'.format(image_path)
    if image_name not in list_of_images:
        raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory(image_directory, image_name)

# general page layout
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

if __name__ == '__main__':

    # set random port to avoid update issues using older sessions
    port = 8000 + np.random.randint(100)*10

    # open in the browser
    webbrowser.open('http://127.0.0.1:'+str(port))

    # dash server
    app.run_server(debug = True, port=port)

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
# Dash Plotly: https://dash.plotly.com/
# Numpy Reference: https://numpy.org/doc/stable/reference/index.html
# Template:    https://dash-bootstrap-components.opensource.faculty.ai/

