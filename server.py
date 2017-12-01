import os
from flask import Flask, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename

import cv2
import numpy as np

UPLOAD_FOLDER = './images'
CONVERTED_FOLDER = './static'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CONVERTED_FOLDER'] = CONVERTED_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify(error = 'No file part')
        file = request.files['file']

        if file.filename == '':
            return jsonify(error = 'No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        filename_no_ext = file.filename.split('.')[0]

        BLUR = 21
        CANNY_THRESH_1 = 1
        CANNY_THRESH_2 = 200
        MASK_DILATE_ITER = 10
        MASK_ERODE_ITER = 10
        MASK_COLOR = (0.0,0.0,1.0)

        img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
        edges = cv2.dilate(edges, None)
        edges = cv2.erode(edges, None)

        contour_info = []
        img2, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            contour_info.append((
                c,
                cv2.isContourConvex(c),
                cv2.contourArea(c),
            ))
        contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
        max_contour = contour_info[0]

        mask = np.zeros(edges.shape)

        cv2.fillConvexPoly(mask, max_contour[0], (255))

        mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
        mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
        mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)


        mask_stack = np.dstack([mask]*3)

        mask_stack  = mask_stack.astype('float32') / 255.0
        img         = img.astype('float32') / 255.0

        masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR)
        masked = (masked * 255).astype('uint8')
        c_red, c_green, c_blue = cv2.split(img)
        img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))

        cv2.imwrite('./static/' + filename_no_ext + '.png',  img_a*255)

    return jsonify(url = os.path.join(app.config['UPLOAD_FOLDER'],filename))

app.run(host= '0.0.0.0')
