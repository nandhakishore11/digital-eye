import os
import keras
import keras.backend as K
from keras.models import load_model
from PIL import Image
import numpy as np
import tensorflow as tf
import picamera
import picamera.array

yolo_model = load_model('tiny_yolo_voc.h5')

def get_image():
	array = np.zeros((1024,1024,3))
	with picamera.PiCamera() as camera:
		with picamera.array.PiRGBArray(camera) as output:
			camera.resolution = (1024, 1024)
			camera.capture(output, 'rgb')
			array = np.array(output.array)
			return Image.fromarray(np.uint8(array))

def filter_predictions(out_xy1, out_xy2,out_conf, score_threshold=0.6):
    return (out_xy1[out_conf > score_threshold], out_xy2[out_conf>score_threshold], out_class_probs[out_conf>score_threshold], out_conf[out_conf>score_threshold])

def nms(xy1, xy2, conf,class_probs, max_boxes,iou_threshold=0.2):
    max_boxes_tensor = tf.Variable(max_boxes, dtype='int32')
    K.get_session().run(tf.global_variables_initializer())
    nms_index = tf.image.non_max_suppression(np.concatenate((xy1[:,::-1], xy2[:,::-1]), axis=1), conf, max_boxes_tensor, iou_threshold)
    out_nms = K.get_session().run([nms_index])
    return (xy1[out_nms], xy2[out_nms], conf[out_nms], class_probs[out_nms])


def get_prediction_text(nms_xy1, nms_class_probs):
    xy1 = nms_xy1.astype('object')
    mask = xy1<208
    xy1[:,0][mask[:,0]] = ' left .'
    xy1[:,1][mask[:,1]] = ' top '
    xy1[:,0][np.invert(mask[:,0])] = ' right .'
    xy1[:,1][np.invert(mask[:,1])] = ' top '
    selected_class = np.argmax(nms_class_probs, axis=1)
    theres_a = np.tile(np.array([' There is a ']), (xy1.shape[0],1))
    to_your =np.tile(np.array([' to your ']), (xy1.shape[0],1))
    out_text = np.concatenate(( theres_a,np.take(voc_classes, selected_class).reshape(xy1.shape[0],1),to_your,xy1[:,::-1]), axis=1)
    return(''.join(np.matrix.flatten(out_text)))

def run_digital_eye():
	feats = yolo_model.output
	feats = K.reshape(feats, [1,13,13,5,25])
	box_xy = K.sigmoid(feats[...,:2])
	box_wh = K.exp(feats[...,2:4])
	box_confidence = K.sigmoid(feats[...,4:5])
	box_class_probs = K.softmax(feats[...,5:])
	box_xy = K.reshape(box_xy, (845,2))
	conv_index = np.zeros((845,2))
	conv_index[:,0] = np.repeat(np.tile(np.arange(13), 13), 5)
	conv_index[:,1]=np.repeat(np.matrix.flatten(np.mgrid[0:13, 0:13][0]), 5)
	box_xy = (box_xy + conv_index) / 13
	out_xy, out_wh, out_conf, out_class_probs = K.get_session().run([box_xy, box_wh, box_confidence, box_class_probs], feed_dict={yolo_model.input : image_data})
	