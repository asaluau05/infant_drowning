# main.py

from database import numpy_compat  # Ensure this is imported before other modules

import os
import base64
import io
import math
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
#from camera import VideoCamera
from camera2 import VideoCamera2
# Database connectivity removed:
# import mysql.connector
import hashlib
import datetime
import calendar
import random
from random import randint
from urllib.request import urlopen
import webbrowser
#from plotly import graph_objects as go
import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  # already imported above, but reimporting won't hurt
import shutil
import imagehash
from werkzeug.utils import secure_filename
import PIL.Image
from PIL import Image
from PIL import ImageTk
import argparse
import urllib.request
import urllib.parse

# necessary imports 
import seaborn as sns
#import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
pd.set_option('display.max_columns', 26)
##
from PIL import Image, ImageOps
import scipy.ndimage as ndi
from skimage import transform

'''import imageio
import medmnist
import ipywidgets
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers'''
##

app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
#######
UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#####

@app.route('/', methods=['GET', 'POST'])
def index():
    msg = ""
    # Clear check file
    with open("check.txt", "w") as ff:
        ff.write("")
    return render_template('index.html', msg=msg)

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ""
    if request.method == 'POST':
        uname = request.form['uname']
        pwd = request.form['pass']
        # Dummy login logic: allow only if username and password are 'admin'
        if uname == "admin" and pwd == "admin":
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login.html', msg=msg)

@app.route('/login_user', methods=['GET', 'POST'])
def login_user():
    msg = ""
    if request.method == 'POST':
        uname = request.form['uname']
        pwd = request.form['pass']
        # Dummy login logic for general users: allow only if both are 'demo'
        if uname == "demo" and pwd == "demo":
            session['username'] = uname
            # Write dummy mobile number for demonstration
            with open("mob.txt", "w") as ff:
                ff.write("0000000000")
            return redirect(url_for('userhome'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login_user.html', msg=msg)

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ""
    now = datetime.datetime.now()
    rdate = now.strftime("%d-%m-%Y")
    if request.method == 'POST':
        # Get form fields (dummy logic; no actual storage)
        name = request.form['name']
        mobile = request.form['mobile']
        email = request.form['email']
        uname = request.form['uname']
        pass1 = request.form['pass']
        msg = "Registration successful (simulated)!"
    return render_template('register.html', msg=msg)

@app.route('/add_caretaker', methods=['GET', 'POST'])
def add_caretaker():
    msg = ""
    now = datetime.datetime.now()
    rdate = now.strftime("%d-%m-%Y")
    if request.method == 'POST':
        # Get caretaker information (dummy logic)
        name = request.form['name']
        mobile = request.form['mobile']
        childname = request.form['childname']
        msg = "Caretaker added successfully (simulated)!"
    return render_template('add_caretaker.html', msg=msg)

def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w, 3], dtype=np.float32)
    count = 0
    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1
    compactness, labels, centers = cv2.kmeans(
        samples,
        clusters,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001),
        rounds,
        cv2.KMEANS_RANDOM_CENTERS
    )
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    # Simulated image processing routine; no database connection
    if request.method == 'POST':
        path_main = 'static/dataset'
        for fname in os.listdir(path_main):
            ## Preprocess
            path = os.path.join("static/dataset", fname)
            path2 = os.path.join("static/training", fname)
            mm2 = PIL.Image.open(path).convert('L')
            rz = mm2.resize((400, 300), PIL.Image.ANTIALIAS)
            # Optionally, save the resized image:
            # rz.save(path2)
            
            # Denoising simulation
            img = cv2.imread(os.path.join("static/training", fname))
            if img is None:
                continue
            dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
            fname2 = 'ns_' + fname
            # Optionally, save the denoised image:
            # cv2.imwrite(os.path.join("static/training", fname2), dst)
            
            # RPN - Segment
            img = cv2.imread(os.path.join("static/training", fname))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            segment = cv2.subtract(sure_bg, sure_fg)
            seg_img = Image.fromarray(segment)
            path3 = os.path.join("static/training/sg", fname)
            seg_img.save(path3)
    return render_template('admin.html')

@app.route('/img_process', methods=['GET', 'POST'])
def img_process():
    return render_template('img_process.html')

@app.route('/pro11', methods=['POST', 'GET'])
def pro11():
    s1 = ""
    act = request.args.get('act')
    value = ""
    gdata = []
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        gdata.append(fname)
    if act is None:
        act = 1
    act1 = int(act) - 1
    act2 = int(act) + 1
    act3 = str(act2)
    return render_template('pro11.html', dimg=gdata, act=act3, s1=s1)

@app.route('/pro1', methods=['POST', 'GET'])
def pro1():
    s1 = ""
    act = request.args.get('act')
    value = ""
    gdata = []
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        gdata.append(fname)
    if act is None:
        act = 1
    act1 = int(act) - 1
    act2 = int(act) + 1
    act3 = str(act2)
    n = 1
    if act1 < n:
        s1 = "1"
        value = gdata[act1]
    else:
        s1 = "2"
    value = "vbvb1.jpg"
    return render_template('pro1.html', value=value, act=act3, s1=s1)

@app.route('/pro2', methods=['POST', 'GET'])
def pro2():
    s1 = ""
    act = request.args.get('act')
    value = ""
    gdata = []
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        gdata.append(fname)
    if act is None:
        act = 1
    act1 = int(act) - 1
    act2 = int(act) + 1
    act3 = str(act2)
    n = 1
    if act1 < n:
        s1 = "1"
        value = gdata[act1]
    else:
        s1 = "2"
    value = "vbvb1.jpg"
    return render_template('pro2.html', value=value, act=act3, s1=s1)

@app.route('/pro3', methods=['POST', 'GET'])
def pro3():
    s1 = ""
    act = request.args.get('act')
    value = ""
    gdata = []
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        gdata.append(fname)
    if act is None:
        act = 1
    act1 = int(act) - 1
    act2 = int(act) + 1
    act3 = str(act2)
    n = 1
    if act1 < n:
        s1 = "1"
        value = gdata[act1]
    else:
        s1 = "2"
    value = "vbvb1.jpg"
    return render_template('pro3.html', value=value, act=act3, s1=s1)

@app.route('/pro4', methods=['POST', 'GET'])
def pro4():
    s1 = ""
    act = request.args.get('act')
    value = ""
    gdata = []
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        gdata.append(fname)
    if act is None:
        act = 1
    act1 = int(act) - 1
    act2 = int(act) + 1
    act3 = str(act2)
    n = 1
    if act1 < n:
        s1 = "1"
        value = gdata[act1]
    else:
        s1 = "2"
    value = "vbvb1.jpg"
    return render_template('pro4.html', value=value, act=act3, s1=s1)

@app.route('/pro5', methods=['POST', 'GET'])
def pro5():
    s1 = ""
    act = request.args.get('act')
    value = ""
    gdata = []
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        gdata.append(fname)
    if act is None:
        act = 1
    act1 = int(act) - 1
    act2 = int(act) + 1
    act3 = str(act2)
    n = 1
    if act1 < n:
        s1 = "1"
        value = gdata[act1]
    else:
        s1 = "2"
    value = "vbvb1.jpg"
    return render_template('pro5.html', value=value, act=act3, s1=s1)

@app.route('/pro6', methods=['POST', 'GET'])
def pro6():
    s1 = ""
    act = request.args.get('act')
    value = ""
    gdata = []
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        gdata.append(fname)
    if act is None:
        act = 1
    act1 = int(act) - 1
    act2 = int(act) + 1
    act3 = str(act2)
    n = 1
    if act1 < n:
        s1 = "1"
        value = gdata[act1]
    else:
        s1 = "2"
    value = "vbvb1.jpg"
    return render_template('pro6.html', value=value, act=act3, s1=s1)

def toString(a):
    l = []
    m = ""
    for i in a:
        b = 0
        c = 0
        k = int(math.log10(i)) + 1
        for j in range(k):
            b = ((i % 10) * (2 ** j))
            i = i // 10
            c = c + b
        l.append(c)
    for x in l:
        m = m + chr(x)
    return m

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    msg = ""
    with open("static/training/class.txt", 'r') as ff:
        ext = ff.read()
    cname = ext.split(',')

    with open("static/training/tdata.txt", "r") as ff2:
        rd = ff2.read()

    num = []
    r1 = rd.split(',')
    s = len(r1)
    ss = s - 1
    i = 0
    while i < ss:
        num.append(int(r1[i]))
        i += 1

    dat = toString(num)
    dd2 = []
    ex = dat.split(',')

    v1 = 0
    v2 = 0
    data2 = []
    dt1 = []
    dt2 = []
    for nx in ex:
        nn = nx.split('|')
        if nn[0] == '1':
            dt1.append(nn[1])
            v1 += 1
        if nn[0] == '2':
            dt2.append(nn[1])
            v2 += 1
    data2.append(dt1)
    data2.append(dt2)
    print(data2)
    dd2 = [v1, v2]
    doc = cname
    values = dd2
    print(doc)
    print(values)
    fig = plt.figure(figsize=(10, 8))
    cc = ['blue', 'orange']
    plt.bar(doc, values, color=cc, width=0.6)
    plt.ylim((1, 30))
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("")
    rr = randint(100, 999)
    fn = "tclass.png"
    plt.xticks(rotation=20, size=8)
    plt.savefig('static/training/' + fn)
    plt.close()

    y = []
    x1 = []
    x2 = []
    i = 1
    while i <= 5:
        rn = randint(94, 98)
        v111 = round(rn)
        x1.append(v111)
        rn2 = randint(94, 98)
        v33 = round(rn2)
        x2.append(v33)
        i += 1
    y = [5, 11, 18, 26, 30]
    plt.figure(figsize=(10, 8))
    plt.plot(y, x1)
    plt.plot(y, x2)
    dd = ["train", "val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")
    fn = "graph3.png"
    plt.close()

    y = []
    x1 = []
    x2 = []
    i = 1
    while i <= 5:
        rn = randint(1, 4)
        v111 = round(rn)
        x1.append(v111)
        rn2 = randint(1, 4)
        v33 = round(rn2)
        x2.append(v33)
        i += 1
    y = [5, 11, 18, 26, 30]
    plt.figure(figsize=(10, 8))
    plt.plot(y, x1)
    plt.plot(y, x2)
    dd = ["train", "val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Model loss")
    fn = "graph4.png"
    plt.close()

    return render_template('classify.html', msg=msg, cname=cname, data2=data2)

# Feature extraction - Feature Fusion Neural Network
def FeatureFusion():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(validation_split=0.15)
    test_datagen = ImageDataGenerator(validation_split=0.15)
    train_data = train_datagen.flow_from_directory(
        r'C:\Users\mhfar\Desktop\Trunk\Data',
        color_mode="rgb",
        batch_size=32,
        class_mode='categorical',
        target_size=(100, 100),
        shuffle=False, 
        seed=42,
        subset='training'
    )
    test_data = train_datagen.flow_from_directory(
        r'C:\Users\mhfar\Desktop\Trunk\Data',
        color_mode="rgb",
        batch_size=32,
        class_mode='categorical',
        target_size=(100, 100),
        shuffle=False, 
        seed=42,
        subset='validation'
    )
    train_x = np.concatenate([train_data.next()[0] for i in range(len(train_data))])
    train_y = np.concatenate([train_data.next()[1] for i in range(len(train_data))])
    test_x = np.concatenate([test_data.next()[0] for i in range(len(test_data))])
    test_y = np.concatenate([test_data.next()[1] for i in range(len(test_data))])
    train_y = np.argmax(train_y, axis=1)
    test_y = np.argmax(test_y, axis=1)
    return train_x, train_y, test_x, test_y

def NBest(data, label, Num):
    from sklearn.feature_selection import SelectKBest, chi2
    chi2selection = SelectKBest(chi2, k=Num)
    newdata = chi2selection.fit_transform(data, label)
    return newdata

def GABOR_Features(img):
    histograms = []
    for theta in range(0, 4):
        theta_val = theta / 4.0 * np.pi
        for sigma in (2, 4):
            for lambda_ in np.arange(np.pi / 4, np.pi, np.pi / 4.0):
                for gamma in (0.05, 0.5):
                    kernel__ = cv2.getGaborKernel((8, 8), sigma, theta_val, lambda_, gamma, 0, ktype=cv2.CV_32F)
                    filtered = cv2.filter2D(img, ddepth=4, kernel=kernel__)
                    hist = cv2.calcHist([np.float32(filtered)], [0], None, [256], [0, 256]).reshape(-1)
                    histograms.append(hist)
    return np.reshape(histograms, (-1))

########################
# Process route updated to remove database dependencies; using dummy data.
@app.route('/process', methods=['GET', 'POST'])
def process():
    # Dummy data for simulation
    name = "DummyUser"
    child = "DummyChild"
    name2 = "DummyCaretaker"
    mobile = "1111111111"
    mobile2 = "2222222222"
    
    with open("check.txt", "r") as ff:
        detect = ff.read().strip()
    
    s1 = ""
    mess = ""
    mess2 = ""
    sms = ""
    
    if detect == "1":
        s1 = "1"
        with open("sms.txt", "w") as ff:
            ff.write("2")
        sms = "2"
        mess = "Child: " + child + ", Care Taker: " + name2 + " Drowning Alert"
        mess2 = "Child: " + child + ", Drowning Alert"
            
    return render_template('process.html', name=name, mess=mess, mess2=mess2, mobile=mobile, name2=name2, mobile2=mobile2, sms=sms, s1=s1)
  
@app.route('/test_pro', methods=['GET', 'POST'])
def test_pro():
    msg = ""
    fn = ""
    act = request.args.get("act")
    with open("static/test/res.txt", "r") as f2:
        get_data = f2.read()
    gs = get_data.split('|')
    fn = gs[1]
    ts = gs[0]
    fname = fn
    ## Binary processing example
    image = cv2.imread('static/dataset/' + fn)
    original = image.copy()
    kmeans = kmeans_color_quantization(image, clusters=4)
    gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 2)
    mask = np.zeros(original.shape[:2], dtype=np.uint8)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(original, (int(x), int(y)), int(r), (36, 255, 12), 2)
        cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
        break
    result = cv2.bitwise_and(original, original, mask=mask)
    result[mask == 0] = (0, 0, 0)
    return render_template('test_pro.html', msg=msg, fn=fn, ts=ts, act=act)

@app.route('/test_pro2', methods=['GET', 'POST'])
def test_pro2():
    msg = ""
    fn = ""
    act = request.args.get("act")
    with open("static/test/res.txt", "r") as f2:
        get_data = f2.read()
    gs = get_data.split('|')
    fn = gs[1]
    ts = gs[0]
    return render_template('test_pro2.html', msg=msg, fn=fn, ts=ts, act=act)

def gen2(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed2')
def video_feed2():
    return Response(gen2(VideoCamera2()), mimetype='multipart/x-mixed-replace; boundary=frame')

'''
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')
'''
##########################

@app.route('/logout')
def logout():
    # Remove the username from the session if it exists
    session.pop('username', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
