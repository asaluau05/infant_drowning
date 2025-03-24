# main.py
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import os
import base64
import io
import math
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
#from camera import VideoCamera
from camera2 import VideoCamera2
import mysql.connector
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
import numpy as np
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
#%matplotlib inline
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
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  charset="utf8",
  database="infant_drowning"

)
app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
#######
UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = { 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#####
@app.route('/', methods=['GET', 'POST'])
def index():
    msg=""
    ff=open("check.txt","w")
    ff.write("")
    ff.close()
    
    return render_template('index.html',msg=msg)


@app.route('/login', methods=['GET', 'POST'])
def login():
    msg=""

    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM id_admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login.html',msg=msg)

@app.route('/login_user', methods=['GET', 'POST'])
def login_user():
    msg = ""
    if request.method == 'POST':
        uname = request.form['uname']
        pwd = request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM id_register WHERE uname = %s AND pass = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:  # Check if the account exists
            mobile = account[2]  # Access mobile only if account exists
            session['username'] = uname
            with open("mob.txt", "w") as ff:
                ff.write(str(mobile))
            return redirect(url_for('userhome'))
        else:
            msg = 'Incorrect username/password!'  # Friendly error message
    return render_template('login_user.html', msg=msg)


@app.route('/register', methods=['GET', 'POST'])
def register():
    msg=""
    

    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
    
    mycursor = mydb.cursor()
    #if request.method=='GET':
    #    msg = request.args.get('msg')
    if request.method=='POST':
        
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        uname=request.form['uname']
        pass1=request.form['pass']

        mycursor.execute("SELECT max(id)+1 FROM id_register")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
                
        sql = "INSERT INTO id_register(id,name,mobile,email,uname,pass) VALUES (%s, %s, %s, %s, %s, %s)"
        val = (maxid,name,mobile,email,uname,pass1)
        mycursor.execute(sql,val)
        mydb.commit()
        msg="success"
        
    return render_template('register.html',msg=msg)

@app.route('/add_caretaker', methods=['GET', 'POST'])
def add_caretaker():
    msg=""
    

    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
    
    mycursor = mydb.cursor()
    #if request.method=='GET':
    #    msg = request.args.get('msg')
    if request.method=='POST':
        
        name=request.form['name']
        mobile=request.form['mobile']
        childname=request.form['childname']
        
        mycursor.execute("SELECT max(id)+1 FROM id_caretaker")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
                
        sql = "INSERT INTO id_caretaker(id,name,mobile,childname) VALUES (%s, %s, %s, %s)"
        val = (maxid,name,mobile,childname)
        mycursor.execute(sql,val)
        mydb.commit()
        msg="success"
        
    return render_template('add_caretaker.html',msg=msg)


def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

@app.route('/admin', methods=['GET', 'POST'])
def admin():

    '''path_main = 'static/t2'
    for fname in os.listdir(path_main):
        print(fname)
        img = cv2.imread('static/t2/'+fname)
        rez = cv2.resize(img, (400, 300))
        cv2.imwrite("static/dataset/"+fname, rez)'''
    if request.method=='POST':
        path_main = 'static/dataset'
        for fname in os.listdir(path_main):
            ##Preprocess
            path="static/dataset/"+fname
            path2="static/training/"+fname
            mm2 = PIL.Image.open(path).convert('L')
            rz = mm2.resize((400,300), PIL.Image.ANTIALIAS)
            #rz.save(path2)
            
            '''img = cv2.imread(path) 
            dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
            path3="static/training/"+fname
            cv2.imwrite(path3, dst)'''
            #noice
            img = cv2.imread('static/training/'+fname) 
            dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
            fname2='ns_'+fname
            #cv2.imwrite("static/training/"+fname2, dst)
            ######
            ##bin
            '''image = cv2.imread('static/training/'+fname)
            original = image.copy()
            kmeans = kmeans_color_quantization(image, clusters=4)

            # Convert to grayscale, Gaussian blur, adaptive threshold
            gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3,3), 0)
            thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,2)
            
            # Draw largest enclosing circle onto a mask
            mask = np.zeros(original.shape[:2], dtype=np.uint8)
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for c in cnts:
                ((x, y), r) = cv2.minEnclosingCircle(c)
                cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
                cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
                break
            
            # Bitwise-and for result
            result = cv2.bitwise_and(original, original, mask=mask)
            result[mask==0] = (0,0,0)

            
            ###cv2.imshow('thresh', thresh)
            ###cv2.imshow('result', result)
            ###cv2.imshow('mask', mask)
            ###cv2.imshow('kmeans', kmeans)
            ###cv2.imshow('image', image)
            ###cv2.waitKey()

            #cv2.imwrite("static/training/bin_"+fname, thresh)'''
            

            ###RPN - Segment
            img = cv2.imread('static/training/'+fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

            # sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=3)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            segment = cv2.subtract(sure_bg,sure_fg)
            img = Image.fromarray(img)
            segment = Image.fromarray(segment)
            path3="static/training/sg/sg_"+fname
            #segment.save(path3)
            ####
           
            '''
            img = cv2.imread(path2)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            # noise removal
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

            # sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=3)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            segment = cv2.subtract(sure_bg,sure_fg)
            img = Image.fromarray(img)
            segment = Image.fromarray(segment)
            path3="static/process2/"+rs[2]
            segment.save(path3)
            '''
            #####
            image = cv2.imread(path2)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edged = cv2.Canny(gray, 50, 100)
            image = Image.fromarray(image)
            edged = Image.fromarray(edged)
            path4="static/training/fg/fg_"+fname
            #edged.save(path4)
        return redirect(url_for('img_process'))

    return render_template('admin.html')

@app.route('/img_process', methods=['GET', 'POST'])
def img_process():
    
    return render_template('img_process.html')

@app.route('/pro11',methods=['POST','GET'])
def pro11():
    s1=""
    
    act = request.args.get('act')
    value=""

    gdata=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        gdata.append(fname)

    if act is None:
        act=1
        
    act1=int(act)-1
    act2=int(act)+1
    act3=str(act2)

    
    return render_template('pro11.html', dimg=gdata,act=act3,s1=s1)

@app.route('/pro1',methods=['POST','GET'])
def pro1():
    s1=""
    
    act = request.args.get('act')
    value=""

    gdata=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        gdata.append(fname)

    if act is None:
        act=1
        
    act1=int(act)-1
    act2=int(act)+1
    act3=str(act2)
    
    n=1
    if act1<n:
        s1="1"
        value=gdata[act1]
    else:
        s1="2"

    value="vbvb1.jpg"
    return render_template('pro1.html', value=value,act=act3,s1=s1)

@app.route('/pro2',methods=['POST','GET'])
def pro2():
    s1=""
    
    act = request.args.get('act')
    value=""

    gdata=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        gdata.append(fname)

    if act is None:
        act=1
        
    act1=int(act)-1
    act2=int(act)+1
    act3=str(act2)
    
    n=1
    if act1<n:
        s1="1"
        value=gdata[act1]
    else:
        s1="2"

    value="vbvb1.jpg"
    return render_template('pro2.html', value=value,act=act3,s1=s1)

@app.route('/pro3',methods=['POST','GET'])
def pro3():
    s1=""
    
    act = request.args.get('act')
    value=""

    gdata=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        gdata.append(fname)

    if act is None:
        act=1
        
    act1=int(act)-1
    act2=int(act)+1
    act3=str(act2)
    
    n=1
    if act1<n:
        s1="1"
        value=gdata[act1]
    else:
        s1="2"

    value="vbvb1.jpg"
    return render_template('pro3.html', value=value,act=act3,s1=s1)

@app.route('/pro4',methods=['POST','GET'])
def pro4():
    s1=""
    
    act = request.args.get('act')
    value=""

    gdata=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        gdata.append(fname)

    if act is None:
        act=1
        
    act1=int(act)-1
    act2=int(act)+1
    act3=str(act2)
    
    n=1
    if act1<n:
        s1="1"
        value=gdata[act1]
    else:
        s1="2"

    value="vbvb1.jpg"
    return render_template('pro4.html', value=value,act=act3,s1=s1)

@app.route('/pro5',methods=['POST','GET'])
def pro5():
    s1=""
    
    act = request.args.get('act')
    value=""

    gdata=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        gdata.append(fname)

    if act is None:
        act=1
        
    act1=int(act)-1
    act2=int(act)+1
    act3=str(act2)
    
    n=1
    if act1<n:
        s1="1"
        value=gdata[act1]
    else:
        s1="2"

    value="vbvb1.jpg"
    return render_template('pro5.html', value=value,act=act3,s1=s1)

@app.route('/pro6',methods=['POST','GET'])
def pro6():
    s1=""
    
    act = request.args.get('act')
    value=""

    gdata=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        gdata.append(fname)

    if act is None:
        act=1
        
    act1=int(act)-1
    act2=int(act)+1
    act3=str(act2)
    
    n=1
    if act1<n:
        s1="1"
        value=gdata[act1]
    else:
        s1="2"

    value="vbvb1.jpg"
    return render_template('pro6.html', value=value,act=act3,s1=s1)

def toString(a):
  l=[]
  m=""
  for i in a:
    b=0
    c=0
    k=int(math.log10(i))+1
    for j in range(k):
      b=((i%10)*(2**j))   
      i=i//10
      c=c+b
    l.append(c)
  for x in l:
    m=m+chr(x)
  return m
@app.route('/classify', methods=['GET', 'POST'])
def classify():
    msg=""
    
    ff=open("static/training/class.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split(',')


    ##    
    ff2=open("static/training/tdata.txt","r")
    rd=ff2.read()
    ff2.close()

    num=[]
    r1=rd.split(',')
    s=len(r1)
    ss=s-1
    i=0
    while i<ss:
        num.append(int(r1[i]))
        i+=1

    #print(num)
    dat=toString(num)
    dd2=[]
    ex=dat.split(',')
    ##

    #ffq=open("static/trained/adata.txt",'r')
    #ext1=ffq.read()
    #ffq.close()

    v1=0
    v2=0
    
    data2=[]
    #ex=ext1.split(',')
    dt1=[]
    dt2=[]
    
    g=0
    for nx in ex:
        g+=1
        nn=nx.split('|')
        if nn[0]=='1':
            
            dt1.append(nn[1])
            
            v1+=1
        if nn[0]=='2':
            dt2.append(nn[1])
            
            v2+=1
       
        
    data2.append(dt1)
    data2.append(dt2)
    
    print(data2)   
    dd2=[v1,v2]
    doc = cname #list(data.keys())
    values = dd2 #list(data.values())
    print(doc)
    print(values)
    fig = plt.figure(figsize = (10, 8))
     
    # creating the bar plot
    cc=['blue','orange']
    plt.bar(doc, values, color =cc,
            width = 0.6)
 

    plt.ylim((1,30))
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("")

    rr=randint(100,999)
    fn="tclass.png"
    plt.xticks(rotation=20,size=8)
    plt.savefig('static/training/'+fn)
    
    plt.close()
    #plt.clf()
    ##
    #graph3########################################
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(94,98)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(94,98)
        v2='0.'+str(rn2)

        
        #v22=float(v2)
        v33=round(rn2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[5,11,18,26,30]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    
    plt.figure(figsize=(10, 8))
    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")
    
    fn="graph3.png"
    #plt.savefig('static/training/'+fn)
    plt.close()
    #######################################################
    #graph4
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,4)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(1,4)
        v2='0.'+str(rn2)

        
        #v22=float(v2)
        v33=round(rn2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[5,11,18,26,30]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    
    plt.figure(figsize=(10, 8))
    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Model loss")
    
    fn="graph4.png"
    #plt.savefig('static/training/'+fn)
    plt.close()
    
    return render_template('classify.html',msg=msg,cname=cname,data2=data2)

###Feature extraction- Feature Fusion Neural Network
def FeatureFusion():
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
        subset='training')

    test_data = train_datagen.flow_from_directory(
        r'C:\Users\mhfar\Desktop\Trunk\Data',
        color_mode="rgb",
        batch_size=32,
        class_mode='categorical',
        target_size=(100, 100),
        shuffle=False, 
        seed=42,
        subset='validation')

    train_x=np.concatenate([train_data.next()[0] for i in range(train_data.__len__())])
    train_y=np.concatenate([train_data.next()[1] for i in range(train_data.__len__())])

    test_x=np.concatenate([test_data.next()[0] for i in range(test_data.__len__())])
    test_y=np.concatenate([test_data.next()[1] for i in range(test_data.__len__())])

    train_y = np.argmax(train_y, axis = 1)
    test_y = np.argmax(test_y, axis = 1)

    return train_x, train_y, test_x, test_y
def NBest(data, label, Num):
    
    chi2selection = SelectKBest(chi2, k=Num)
    newdata = chi2selection.fit_transform(data, label)
    return newdata
def GABOR_Features(img):
    histograms = []
    for theta in range(0, 4):
        theta = deepcopy(theta/4. * np.pi)
        for sigma in (2, 4):
            for lambda_ in np.arange(np.pi / 4, np.pi, np.pi / 4.):
                for gamma in (0.05, 0.5):
                    kernel__ = cv.getGaborKernel((8, 8), sigma, theta, lambda_, gamma, 0, ktype=cv.CV_32F)
                    filtered = cv.filter2D(img, ddepth=4, kernel= kernel__)
                    hist = cv.calcHist([np.float32(filtered)],[0],None,[256],[0,256]).reshape(-1)
                    histograms.append(hist)
    return np.reshape(histograms, (-1))
##
#Classification
def CNN():
    #Lets start by loading the Cifar10 data
    (X, y), (X_test, y_test) = cifar10.load_data()

    #Keep in mind the images are in RGB
    #So we can normalise the data by diving by 255
    #The data is in integers therefore we need to convert them to float first
    X, X_test = X.astype('float32')/255.0, X_test.astype('float32')/255.0

    #Then we convert the y values into one-hot vectors
    #The cifar10 has only 10 classes, thats is why we specify a one-hot
    #vector of width/class 10
    y, y_test = u.to_categorical(y, 10), u.to_categorical(y_test, 10)

    #Now we can go ahead and create our Convolution model
    model = Sequential()
    #We want to output 32 features maps. The kernel size is going to be
    #3x3 and we specify our input shape to be 32x32 with 3 channels
    #Padding=same means we want the same dimensional output as input
    #activation specifies the activation function
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same',
                     activation='relu'))
    #20% of the nodes are set to 0
    model.add(Dropout(0.2))
    #now we add another convolution layer, again with a 3x3 kernel
    #This time our padding=valid this means that the output dimension can
    #take any form
    model.add(Conv2D(32, (3, 3), activation='relu', padding='valid'))
    #maxpool with a kernet of 2x2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #In a convolution NN, we neet to flatten our data before we can
    #input it into the ouput/dense layer
    model.add(Flatten())
    #Dense layer with 512 hidden units
    model.add(Dense(512, activation='relu'))
    #this time we set 30% of the nodes to 0 to minimize overfitting
    model.add(Dropout(0.3))
    #Finally the output dense layer with 10 hidden units corresponding to
    #our 10 classe
    model.add(Dense(10, activation='softmax'))
    #Few simple configurations
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(momentum=0.5, decay=0.0004), metrics=['accuracy'])
    #Run the algorithm!
    model.fit(X, y, validation_data=(X_test, y_test), epochs=25,
              batch_size=512)
    #Save the weights to use for later
    model.save_weights("cifar10.hdf5")
    #Finally print the accuracy of our model!
    print("Accuracy: &2.f%%" %(model.evaluate(X_test, y_test)[1]*100))


##################
#Live Monitoring - Video Vision Transformer
def preprocess():
    """Preprocess the frames tensors and parse the labels."""
    # Preprocess images
    frames = tf.image.convert_image_dtype(
        frames[
            ..., tf.newaxis
        ],  # The new axis is to help for further processing with Conv3D layers
        tf.float32,
    )
    # Parse label
    label = tf.cast(label, tf.float32)
    return frames, label


    def prepare_dataloader(
        videos: np.ndarray,
        labels: np.ndarray,
        loader_type: str = "train",
        batch_size: int = BATCH_SIZE,
    ):
        """Utility function to prepare the dataloader."""
        dataset = tf.data.Dataset.from_tensor_slices((videos, labels))

        if loader_type == "train":
            dataset = dataset.shuffle(BATCH_SIZE * 2)

        dataloader = (
            dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        return dataloader
    def create_vivit_classifier(
        tubelet_embedder,
        positional_encoder,
        input_shape=INPUT_SHAPE,
        transformer_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        embed_dim=PROJECTION_DIM,
        layer_norm_eps=LAYER_NORM_EPS,
        num_classes=NUM_CLASSES,
    ):
        # Get the input layer
        inputs = layers.Input(shape=input_shape)
        # Create patches.
        patches = tubelet_embedder(inputs)
        # Encode patches.
        encoded_patches = positional_encoder(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(transformer_layers):
            # Layer normalization and MHSA
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
            )(x1, x1)

            # Skip connection
            x2 = layers.Add()([attention_output, encoded_patches])

            # Layer Normalization and MLP
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = keras.Sequential(
                [
                    layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                    layers.Dense(units=embed_dim, activation=tf.nn.gelu),
                ]
            )(x3)

            # Skip connection
            encoded_patches = layers.Add()([x3, x2])

        # Layer normalization and Global average pooling.
        representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
        representation = layers.GlobalAvgPool1D()(representation)

        # Classify outputs.
        outputs = layers.Dense(units=num_classes, activation="softmax")(representation)

        # Create the Keras model.
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
#######
@app.route('/userhome', methods=['GET', 'POST'])
def userhome():
    msg=""
    uname=""
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()

    ff=open("check.txt","w")
    ff.write("")
    ff.close()

    f1=open("file.txt","w")
    f1.write("")
    f1.close()
    
    mycursor.execute("SELECT * FROM id_register where uname=%s",(uname,))
    data = mycursor.fetchone()

        
    return render_template('userhome.html',msg=msg,data=data)

@app.route('/update', methods=['GET', 'POST'])
def update():
    msg=""
    uname=""
    act=request.args.get("act")
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()

    f1=open("file.txt","w")
    f1.write("")
    f1.close()
    
    mycursor.execute("SELECT * FROM id_register where uname=%s",(uname,))
    data = mycursor.fetchone()
    if request.method=='POST':
        '''mobile=request.form['mobile']
        mobile2=request.form['mobile2']

        mycursor.execute("update id_register set mobile=%s,caretaker_mobile=%s where uname=%s",(mobile,mobile2,uname))
        mydb.commit()
        msg="ok"'''
        name=request.form['name']
        mobile=request.form['mobile']
        childname=request.form['childname']

        mycursor.execute("update id_caretaker set status=0 where status=1")
        mydb.commit()
        
        mycursor.execute("SELECT max(id)+1 FROM id_caretaker")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
                
        sql = "INSERT INTO id_caretaker(id,name,mobile,childname,uname,status) VALUES (%s, %s, %s, %s, %s, %s)"
        val = (maxid,name,mobile,childname,uname,'1')
        mycursor.execute(sql,val)
        mydb.commit()
        msg="ok"

    mycursor.execute("SELECT * FROM id_caretaker where uname=%s",(uname,))
    data1 = mycursor.fetchall()

    if act=="del":
        did=request.args.get("did")
        mycursor.execute("delete from id_caretaker where id=%s",(did,))
        mydb.commit()
        return redirect(url_for('update'))

    if act=="yes":
        sid=request.args.get("sid")
        mycursor.execute("update id_caretaker set status=0 where status=1")
        mydb.commit()
        mycursor.execute("update id_caretaker set status=1 where id=%s",(sid,))
        mydb.commit()
        return redirect(url_for('update'))

        
    return render_template('update.html',msg=msg,act=act,data=data,data1=data1)


@app.route('/test_video', methods=['GET', 'POST'])
def test_video():
    msg=""
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()

       
    mycursor.execute("SELECT * FROM id_register where uname=%s",(uname,))
    data = mycursor.fetchone()

    ff=open("sms.txt","w")
    ff.write("1")
    ff.close()

    f1=open("file.txt","r")
    st=f1.read()
    f1.close()
        

    vdata=[]
    path_main = 'static/videos/'
    for fname in os.listdir(path_main):
        vdata.append(fname)

    if request.method=='POST':
        
        video=request.form['video']
        

        f1=open("file.txt","w")
        f1.write("static/videos/"+video)
        f1.close()

        ff=open("people.txt","w")
        ff.write("0")
        ff.close()

        return redirect(url_for('test_video'))

    return render_template('test_video.html',msg=msg,vdata=vdata,st=st)

@app.route('/detection', methods=['GET', 'POST'])
def detection():
    msg=""

    return render_template('detection.html',msg=msg)

@app.route('/process',methods=['POST','GET'])
def process():
    msg=""
    s1=""
    mess=""
    mess2=""
    mobile=""
    sms=""
   

    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    
    mycursor.execute("SELECT * FROM id_register where uname=%s",(uname,))
    data = mycursor.fetchone()

    
    name=data[1]
    mobile=data[2]

    mycursor.execute("SELECT * FROM id_caretaker where uname=%s && status=1",(uname,))
    data2 = mycursor.fetchone()
    name2=data2[1]
    mobile2=data2[2]
    child=data2[3]
    
    ff=open("check.txt","r")
    detect=ff.read()
    ff.close()

    ff=open("sms.txt","r")
    sms=ff.read()
    ff.close()

    if detect=="1":
        s1="1"

        ff=open("sms.txt","w")
        ff.write("2")
        ff.close()

        mess="Child: "+child+",Care Taker:"+name2+" Drowning Alert"
        mess2="Child: "+child+", Drowning Alert"
        

            
    return render_template('process.html',name=name,mess=mess,mess2=mess2,mobile=mobile,name2=name2,mobile2=mobile2,sms=sms,s1=s1)


  
@app.route('/test_pro', methods=['GET', 'POST'])
def test_pro():
    msg=""
    fn=""
    act=request.args.get("act")
    f2=open("static/test/res.txt","r")
    get_data=f2.read()
    f2.close()

    gs=get_data.split('|')
    fn=gs[1]
    ts=gs[0]
    fname=fn
    ##bin
    image = cv2.imread('static/dataset/'+fn)
    original = image.copy()
    kmeans = kmeans_color_quantization(image, clusters=4)

    # Convert to grayscale, Gaussian blur, adaptive threshold
    gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,2)

    # Draw largest enclosing circle onto a mask
    mask = np.zeros(original.shape[:2], dtype=np.uint8)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
        cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
        break
    
    # Bitwise-and for result
    result = cv2.bitwise_and(original, original, mask=mask)
    result[mask==0] = (0,0,0)

    
    ###cv2.imshow('thresh', thresh)
    ###cv2.imshow('result', result)
    ###cv2.imshow('mask', mask)
    ###cv2.imshow('kmeans', kmeans)
    ###cv2.imshow('image', image)
    ###cv2.waitKey()

    #cv2.imwrite("static/upload/bin_"+fname, thresh)
    

    ###fg
    img = cv2.imread('static/dataset/'+fn)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    segment = cv2.subtract(sure_bg,sure_fg)
    img = Image.fromarray(img)
    segment = Image.fromarray(segment)
    path3="static/upload/test/fg_"+fname
    #segment.save(path3)
    
        
    return render_template('test_pro.html',msg=msg,fn=fn,ts=ts,act=act)

@app.route('/test_pro2', methods=['GET', 'POST'])
def test_pro2():
    msg=""
    fn=""
    act=request.args.get("act")
    f2=open("static/test/res.txt","r")
    get_data=f2.read()
    f2.close()

    gs=get_data.split('|')
    fn=gs[1]
    ts=gs[0]
    return render_template('test_pro2.html',msg=msg,fn=fn,ts=ts,act=act)
###
def gen2(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



@app.route('/video_feed2')
def video_feed2():
    return Response(gen2(VideoCamera2()), mimetype='multipart/x-mixed-replace; boundary=frame')
########################


'''def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')
'''
########################


##########################
@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


