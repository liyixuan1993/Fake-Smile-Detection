# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 19:42:52 2017

@author: kcchiu
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import sys
import shutil

folder_fake = 'data/fake_smile'
folder_real = 'data/real_smile'

folder_fake_faces = 'data/faces/fake_smile'
folder_real_faces = 'data/faces/real_smile'

folder_results_fake_test  = 'data/pred_results/fake_smile/test'
folder_results_real_test  = 'data/pred_results/real_smile/test'

detect_face = 0                                                                 # 0: using the existing cropped face images. 1: use face detectFace to crop the faces out of the original image

def load_images_from_folder(folder):
    '''
    Input:
        folder             :The path of the image folder.
    Output:
        images             :A (N,) numpy array, where N is the total number of 
                            images.
    '''
    images = []
    for filename in os.listdir(folder):
        if filename.startswith('.') or filename == 'Thumbs.db':
            continue
        img = Image.open(os.path.join(folder,filename)).convert('L')            # open the image and convert to uint8 (gray scale)
        img = np.array(img)                                                     # convert image to a numpy array
        if img is not None:
            images.append(img)
    images = np.array(images)
    return images

def resize_image(images, size):
    '''
    Input: 
        images             :An object of shape (N,). N is the total number of 
                            images.
        size               :A ndarray (rows, cols) of the output image shape.
    Output:
        resized_images     :An (N,rows,cols) numpy array.
    '''
    resized_images = []
    for i,j in enumerate(images):
        resized_images.append(resize(images[i], ((size[0]),size[1]),            # resize the image
                                     mode='reflect'))
    resized_images = np.array(resized_images)                                   # convert image to a numpy array
    return resized_images

def detectFace(img):
    '''
    Input:
        img                 : A gray scale image represented by numpy array.
    Output:
        bbox                : The four corners of bounding boxes for all 
                              detected faces in numpy arrray of shape (number 
                              of detected faces,4,2).
                                ++++++++++++++++++
                                +(x0,y0)  (x1,y1)+
                                +                +
                                +                +
                                +      bbox      +
                                +                +
                                +                +
                                +(x2,y2)  (x3,y3)+
                                ++++++++++++++++++
    '''
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    delta_scal = 0
    while True:
        faces = face_cascade.detectMultiScale(img, 3.0-delta_scal, 2)
        if len(faces) != 0:                                                     # at least one face is detected
            break
        else:                                                                   # no face detected, re-detecting with new parameters...
            if 3.0-delta_scal > 1.01:
                delta_scal += 0.01
            else:
                break
              
    bbox = np.zeros([len(faces),4,2])
        
    for i, (x,y,w,h) in enumerate(faces):
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)                          # draw a red rectangle around the face  
        bbox[i,:,:] = np.array([[y,x],[y,x+w],[y+h,x],[y+h,x+w]])
        
    if bbox.shape[0] != 1:
        print("\nWarning! Multiple faces detected! Image shown below...")
        plt.figure()
        plt.imshow(img, cmap='gray')
    
    return bbox[0,:,:].astype(int)

new_size = [300,300]                                                            # the output size of all the images regardless of it's original shape

if detect_face:
    print("Loading original images..")
    images = load_images_from_folder(folder_fake)                               # load images from folder
    X_fake = resize_image(images,new_size)                                      # resize to 300*300
    
    images = load_images_from_folder(folder_real)
    X_real = resize_image(images,new_size)
    
    X_fake_f = np.zeros([X_fake.shape[0],new_size[0],new_size[1]])
    X_real_f = np.zeros([X_real.shape[0],new_size[0],new_size[1]])
    
    toolbar_width = 1                                                           # not important, just for printing purpose...
    sys.stdout.write("%s\r" % (" " * toolbar_width))                            # not important, just for printing purpose...
    sys.stdout.flush()                                                          # not important, just for printing purpose...
    sys.stdout.write("\b" * (toolbar_width+1))                                  # not important, just for printing purpose...
    
    shutil.rmtree(folder_fake_faces, ignore_errors=True)                        # remove the folder if it already exists
    os.makedirs(folder_fake_faces)                                              # create a new folder
    shutil.rmtree(folder_real_faces, ignore_errors=True)                        # remove the folder if it already exists
    os.makedirs(folder_real_faces)                                              # create a new folder
    
    for i in range(X_fake.shape[0]):
        bbox = detectFace((255*X_fake[i,:,:]).astype('uint8'))                  # detects the face of the original image
        face_i = X_fake[i,bbox[0,0]:bbox[2,0],bbox[0,1]:bbox[1,1]]              # crop out the face from the original image
        X_fake_f[i,:,:] = resize(face_i, (new_size[0],new_size[1]),
                                 mode='reflect')                                # resize to 300*300
        plt.imsave(folder_fake_faces+"/fake_smile_"+str(i)+".jpg",
                   X_fake_f[i,:,:], cmap='gray')                                # save the detected face
        
        sys.stdout.write("\rDetecting and saving fake smile faces...%.1f%%" % 
                         (i*100/(X_fake.shape[0]-1)))                           # not important, just for printing purpose...
        sys.stdout.flush()                                                      # not important, just for printing purpose...
        
    sys.stdout.write("\n")
        
    for i in range(X_real.shape[0]):
        bbox = detectFace((255*X_real[i,:,:]).astype('uint8'))                  # detects the face of the original image
        face_i = X_real[i,bbox[0,0]:bbox[2,0],bbox[0,1]:bbox[1,1]]              # crop out the face from the original image
        X_real_f[i,:,:] = resize(face_i, (new_size[0],new_size[1]), 
                                 mode='reflect')                                # resize to 300*300
        plt.imsave(folder_real_faces+"/real_smile"+str(i)+".jpg",
                   X_real_f[i,:,:], cmap='gray')                                # save the detected face
        
        sys.stdout.write("\rDetecting and saving real smile faces...%.1f%%" % 
                         (i*100/(X_real.shape[0]-1)))                           # not important, just for printing purpose...
        sys.stdout.flush()                                                      # not important, just for printing purpose...
    
    sys.stdout.write("\n")                                                      # not important, just for printing purpose...
else:
    print("Loading pre-cropped faces from folder...")
    images = load_images_from_folder(folder_fake_faces)                         # load images from folder
    X_fake_f = resize_image(images,new_size)                                    # resize to 300*300
    
    images = load_images_from_folder(folder_real_faces)                         # load images from folder
    X_real_f = resize_image(images,new_size)                                    # resize to 300*300
    
y_fake = np.ones([X_fake_f.shape[0]])                                           # fake smile labels: 1
y_real = np.zeros([X_real_f.shape[0]])                                          # real smile labels: 0

print("Training CNN...")

def reset_graph(seed=42):
    tf.reset_default_graph()
#    tf.set_random_seed(seed)
#    np.random.seed(seed)                                                       # uncomment if want to make train test split return the same samples every time

height = new_size[0]
width = new_size[1]
channels = 1
n_inputs = height * width

conv11_fmaps = 64
conv11_ksize = 3
conv11_stride = 1
conv11_pad = "SAME"

conv21_fmaps = 128
conv21_ksize = 3
conv21_stride = 1
conv21_pad = "SAME"

conv31_fmaps = 256
conv31_ksize = 3
conv31_stride = 1
conv31_pad = "SAME"

conv32_fmaps = 256
conv32_ksize = 3
conv32_stride = 1
conv32_pad = "SAME"

conv41_fmaps = 512
conv41_ksize = 3
conv41_stride = 1
conv41_pad = "SAME"

conv42_fmaps = 512
conv42_ksize = 3
conv42_stride = 1
conv42_pad = "SAME"

conv51_fmaps = 512
conv51_ksize = 3
conv51_stride = 1
conv51_pad = "SAME"

conv52_fmaps = 512
conv52_ksize = 3
conv52_stride = 1
conv52_pad = "SAME"


#pool1_fmaps = conv21_fmaps
#pool2_fmaps = conv31_fmaps
#pool3_fmaps = conv41_fmaps
#pool4_fmaps = conv51_fmaps
pool5_fmaps = conv52_fmaps

n_fc1 = 4
n_fc2 = 4
n_fc3 = 1
n_outputs = 2

reset_graph()

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, height, width], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")

conv11 = tf.layers.conv2d(X_reshaped, filters=conv11_fmaps, 
                         kernel_size=conv11_ksize,
                         strides=conv11_stride, padding=conv11_pad,
                         activation=tf.nn.relu, name="conv11")


with tf.name_scope("pool1"):
    pool1 = tf.nn.max_pool(conv11, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                           padding="VALID")
    
with tf.name_scope("conv21"):
    conv21 = tf.layers.conv2d(pool1, filters=conv21_fmaps, kernel_size=conv21_ksize,
                         strides=conv21_stride, padding=conv21_pad,
                         activation=tf.nn.relu, name="conv21")
#    print("conv21", conv21)
#    tf.get_variable('conv21/conv21', shape=[1, 74, 74, 64])
      
with tf.name_scope("pool2"):
    pool2 = tf.nn.max_pool(conv21, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                           padding="VALID")

with tf.name_scope("conv31"):
    conv31 = tf.layers.conv2d(pool2, filters=conv31_fmaps, kernel_size=conv31_ksize,
                         strides=conv31_stride, padding=conv31_pad,
                         activation=tf.nn.relu, name="conv31")
    
with tf.name_scope("conv32"):
    conv32 = tf.layers.conv2d(conv31, filters=conv32_fmaps, kernel_size=conv32_ksize,
                         strides=conv32_stride, padding=conv32_pad,
                         activation=tf.nn.relu, name="conv32")
        
with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv32, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                           padding="VALID")
    
with tf.name_scope("conv41"):
    conv41 = tf.layers.conv2d(pool3, filters=conv41_fmaps, kernel_size=conv41_ksize,
                         strides=conv41_stride, padding=conv41_pad,
                         activation=tf.nn.relu, name="conv41")
    
with tf.name_scope("conv42"):
    conv42 = tf.layers.conv2d(conv41, filters=conv42_fmaps, kernel_size=conv42_ksize,
                         strides=conv42_stride, padding=conv42_pad,
                         activation=tf.nn.relu, name="conv42")
       
with tf.name_scope("pool4"):
    pool4 = tf.nn.max_pool(conv42, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                           padding="VALID")
    
with tf.name_scope("conv51"):
    conv51 = tf.layers.conv2d(pool4, filters=conv51_fmaps, kernel_size=conv51_ksize,
                         strides=conv51_stride, padding=conv51_pad,
                         activation=tf.nn.relu, name="conv51")
    
with tf.name_scope("conv52"):
    conv52 = tf.layers.conv2d(conv51, filters=conv52_fmaps, kernel_size=conv52_ksize,
                         strides=conv52_stride, padding=conv52_pad,
                         activation=tf.nn.relu, name="conv52")
  
with tf.name_scope("pool5"):
    pool5 = tf.nn.max_pool(conv52, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                           padding="VALID")
#    pool2 = tf.nn.avg_pool(conv52, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
#                           padding="VALID")
    
    print("pool5",pool5)
    print pool5.shape
    pool5_flat = tf.reshape(pool5, shape=[-1, pool5_fmaps * 9 * 9])
    
with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool5_flat, n_fc1, activation=tf.nn.relu, name="fc1")
    
with tf.name_scope("fc2"):
    fc2 = tf.layers.dense(fc1, n_fc2, activation=tf.nn.relu, name="fc2")
    
with tf.name_scope("fc3"):
    fc3 = tf.layers.dense(fc2, n_fc3, activation=tf.nn.relu, name="fc3")

with tf.name_scope("output"):
    logits = tf.layers.dense(fc3, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, 
                                                              labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

X_raw = np.r_[X_fake_f, X_real_f]                                               # concat fake and real smile samples together
y_raw = np.r_[y_fake, y_real]                                                   # 1: fake, 0: real
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, 
                                                    test_size=0.3)              # split randomly into train and test datasets

n_epochs = 100                                                                    # number of epochs to train this model. The larger the better
num_of_batches = 5                                                              # split the training data into batches to avoid insufficient memory error
batch_size = int(X_train.shape[0] / num_of_batches)                             # number of samples in each batch

#filter_summary = tf.summary.image('conv21',conv21)

with tf.Session() as sess:
#    summary_writer = tf.summary.FileWriter('/tmp/logs', sess.graph_def)
    init.run()
    
    for epoch in range(n_epochs):
        acc_train = 0
        acc_test = 0
        
        for batch in range(num_of_batches):                                     # feed in the training data one batch at a time
            from_i = batch*batch_size
            to_i = (batch+1)*batch_size
            
            if batch != num_of_batches-1:                                       # not last batch?
                sess.run(training_op, feed_dict={X: X_train[from_i:to_i], 
                                                 y: y_train[from_i:to_i]})
            else:                                                               # last batch
                sess.run(training_op, feed_dict={X: X_train[from_i:], 
                                                 y: y_train[from_i:]})
                
            acc_train += accuracy.eval(feed_dict={X: X_train[from_i:to_i], 
                                     y: y_train[from_i:to_i]}) / num_of_batches
            acc_test  += accuracy.eval(feed_dict={X: X_test, 
                                                  y: y_test}) / num_of_batches
            
        pred_fake_test = sess.run(Y_proba, feed_dict={X: X_test, y: y_test})    # get the output (Y_proba) of the model
        pred_fake_indices_test = np.where(pred_fake_test[:,1]>=0.5)             # get the indices of the samples that is predicted "fake". If Y_proba >= 0.5, then it is predicted "fake"
        pred_real_indices_test = np.where(pred_fake_test[:,1]<0.5)              # get the indices of the samples that is predicted "real".if Y_proba < 0.5, then it is predicted "real"
        
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
         
    print("Saving the results...")
    
#    summary_writer.add_summary(filter_summary)
    
    shutil.rmtree(folder_results_fake_test, ignore_errors=True)                 # remove the folder if it already exists
    os.makedirs(folder_results_fake_test)                                       # create a new folder
    shutil.rmtree(folder_results_real_test, ignore_errors=True)                 # remove the folder if it already exists
    os.makedirs(folder_results_real_test)                                       # create a new folder
    
    for i,j in np.ndenumerate(pred_fake_indices_test):                          # save all the predicted "fake" samples into the folder
        plt.imsave(folder_results_fake_test+"/fake_smile_pred_test_"+str(i[1])
                                            +".jpg",X_test[j,:,:], cmap='gray')
    for i,j in np.ndenumerate(pred_real_indices_test):                          # save all the predicted "real" samples into the folder
        plt.imsave(folder_results_real_test+"/real_smile_pred_test_"+str(i[1])
                                            +".jpg",X_test[j,:,:], cmap='gray')

#    with tf.variable_scope('conv21', reuse=True) as scope_conv:
#        W_conv1 = tf.get_variable('conv21', shape=[1, 74, 74, 64])
#        weights = W_conv1.eval()
##        print(weights, type(weights), weights.shape)
#        plt.figure()
#        plt.imshow(weights[0,:,:,1])
#        plt.figure()
#        plt.imshow(weights[0,:,:,32])
#        plt.figure()
#        plt.imshow(weights[0,:,:,63])
#        
#    with tf.variable_scope('conv52', reuse=True) as scope_conv:
#        W_conv1 = tf.get_variable('conv52', shape=[1, 2, 2, 512])
#        weights = W_conv1.eval()
##        print(weights, type(weights), weights.shape)
#        plt.figure()
#        plt.imshow(weights[0,:,:,1])
#        plt.imshow(weights[0,:,:,100])
#        plt.imshow(weights[0,:,:,500])
