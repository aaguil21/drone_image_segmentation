import pandas as pd
import numpy as np
import cv2
import tensorflow as tf


def create_image_array(path, file_names: list, size=224):
    images = []
    for file in file_names:
        img = cv2.imread(path+file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation = cv2.INTER_NEAREST )
        images.append(img)
        
    return images

def create_color_map():
    color_map = pd.read_csv('./semantic_drone_dataset/class_dict.csv')
    dummy_color_map = pd.get_dummies(color_map, columns = ['name'], prefix='', prefix_sep='')
    cm = color_map.iloc[:, 1:].T.to_dict()
    cm_id = {tuple(v.values()): k for k, v in cm.items()}
    cm_rgb = {int(k): tuple(v.values()) for k, v in cm.items()}
    
    return cm_id, cm_rgb


def mask_classes(mask):
    shape = list(mask.shape[:2]) + [1]
    mask_class = np.zeros(shape, dtype=int)
    cm_id, cm_rgb = create_color_map()
    sub = lambda x: cm_id[tuple(x)]
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            mask_class[x,y,0] = sub(mask[x,y,:])
    return mask_class


def mask_rgb(mask):
    shape = list(mask.shape[:2]) + [3]
    mask_class = np.zeros(shape, dtype=int)
    cm_id, cm_rgb = create_color_map()
    sub = lambda x: cm_rgb[int(x)]
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            mask_class[x,y,:] = sub(mask[x,y,0])
    return mask_class

def mask_encoding(masks, size=224):
    oh_masks = []
    out_shape = (size, size, 24)
    for mask in masks:
        oh_mask = tf.one_hot(mask, depth=24, axis=3)
        oh_mask = tf.reshape(oh_mask, out_shape)
        oh_masks.append(oh_mask)

    return oh_masks

def mask_decoding(masks, size=224):
    class_masks = []
    out_shape = (size, size, 1)
    for mask in masks:
        class_mask = tf.argmax(tf.reshape(mask, (size, size, 24)), axis=2)
        class_mask = tf.reshape(class_mask, out_shape)
        class_masks.append(class_mask)

    return class_masks

if __name__ == '__main__':
    pass
    