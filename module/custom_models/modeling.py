# Uncomment for use in Google Colab
#!unzip processed.zip -d ./semantic_drone_dataset
#!pip install git+https://github.com/tensorflow/examples.git

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np
import cv2
import albumentations as A
from sklearn.model_selection import train_test_split
import pandas as pd

from tensorflow_examples.models.pix2pix import pix2pix

from IPython.display import clear_output
import matplotlib.pyplot as plt
from glob import glob
import os



transform = [
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Sharpen()
]

# Uncomment for use in Google Colab
#img_folder = './semantic_drone_dataset/processed/images/'
#mask_folder = './semantic_drone_dataset/processed/label_images/'

# Comment out when running notebook in Google Colab
img_folder = '../semantic_drone_dataset/processed/images/'
mask_folder = '../semantic_drone_dataset/processed/label_images/'

img_files = np.sort(os.listdir(img_folder))
mask_files = np.sort(os.listdir(mask_folder))

# Train and Validation data genrator instances
train_generator = DataGenerator(img_folder, mask_folder, img_files, mask_files,batch_size=16, augment=transform, dim=[224, 224] ,shuffle=True, validation_split=0.2, subset='Train')
val_generator = DataGenerator(img_folder, mask_folder, img_files, mask_files,batch_size=16, augment=transform, dim=[224, 224] ,shuffle=True, validation_split=0.2, subset='Valid')


def mask_encoding(masks, size = 224):
    """
    One Hot encode the class value for mask images
    """
    oh_masks = []
    out_shape = (size, size, 24)
    for mask in masks:
        oh_mask = tf.one_hot(mask, depth=24, axis=3)
        oh_mask = tf.reshape(oh_mask, out_shape)
        oh_masks.append(oh_mask)

    return oh_masks

# %%
def mask_decoding(masks, size = 224):
    """
    Undo the one hot encoding for masks. Used for creating images from model predictions. 
    """
    class_masks = []
    out_shape = (size, size, 1)
    for mask in masks:
        class_mask = tf.argmax(tf.reshape(mask, (size, size, 24)), axis=2)
        class_mask = tf.reshape(class_mask, out_shape)
        class_masks.append(class_mask)

    return class_masks


def dice_coef(y_true, y_pred, smooth=10):        
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def dice_coef_multilabel(y_true, y_pred, M=24, smooth=10):
    dice = 0
    for index in range(M):
        dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index], smooth)
    return dice

def dice_coef_multilabel_loss(y_true, y_pred):
    return 1 - dice_coef_multilabel(y_true, y_pred)



def display(display_list, name='None', epoch=0):
  """
  Function to display input image, mask and mask prediction. Will be used in following callback to show model 
  improvements during training. Saves the output graph. 
  """
  fig = plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  file_name = name + '_' + str(epoch) + '_' + 'pred_mask.png' 
  fig.savefig('./imgs/' + file_name)
  plt.show()

  
# Create a sample image and mask the will be referenced in 
im, mk = train_generator.__getitem__(1)
sample_image, sample_mask = im[0], mask_decoding(mk)[0]
display([sample_image, sample_mask])


class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    if epoch % 10 == 0:
      pred = self.model.predict(sample_image[tf.newaxis, ...])
      pred = mask_decoding(pred)[0]
      clear_output(wait=True)
      display([sample_image, sample_mask, pred], self.model.name, epoch)
      print ('\nSample Prediction after epoch {}\n'.format(epoch+1))



OUTPUT_CLASSES = 24


model = unet_model(output_channels=OUTPUT_CLASSES, name='dice_seg')
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=dice_coef_multilabel_loss,
              metrics=['acc',
                       tf.keras.metrics.MeanIoU(num_classes=24, sparse_y_pred=False, sparse_y_true=False)])

# %%
EPOCHS = 51

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('seg_model_dice.h5', 
                                                   save_best_only=True,
                                                   save_weights_only=True)

model_history = model.fit(train_generator, epochs=EPOCHS,
                          validation_data=val_generator,
                          callbacks=[DisplayCallback(), checkpoint_cb])

# %%
fig, ax = plt.subplots(1, 3, figsize=(14,4))
ax[0].plot(model_history.history['loss'], label='Training Loss')
ax[0].plot(model_history.history['val_loss'], label='Validation Loss')
ax[0].legend()

ax[1].plot(model_history.history['acc'], label='Training Accuracy')
ax[1].plot(model_history.history['val_acc'], label='Validation Accuracy')
ax[1].legend()

ax[2].plot(model_history.history['mean_io_u'], label='Training Mean IoU')
ax[2].plot(model_history.history['val_mean_io_u'], label='Validation Mean IoU')
ax[2].legend();
plt.savefig('./imgs/dice_seg_training_curves.png');

# %% [markdown]
# From the training curves, we can see that the model begins to overfit the training data at about 20 epochs. The Mean IoU score does reach above 0.25, which is a relatively low score. 

# %% [markdown]
# Categorical Crossentropy Loss Model
# 

# %%
OUTPUT_CLASSES = 24

model2 = unet_model(output_channels=OUTPUT_CLASSES, name = 'cc_seg')
model2.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['acc',
                       tf.keras.metrics.MeanIoU(num_classes=24, sparse_y_pred=False, sparse_y_true=False)])

# %%
EPOCHS = 51

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('seg_model_cc.h5', 
                                                   save_best_only=True,
                                                   save_weights_only=True)

model2_history = model2.fit(train_generator, epochs=EPOCHS,
                          validation_data=val_generator,
                          callbacks=[DisplayCallback(), checkpoint_cb])

# %%
fig, ax = plt.subplots(1, 3, figsize=(14,4))
ax[0].plot(model2_history.history['loss'], label='Training Loss')
ax[0].plot(model2_history.history['val_loss'], label='Validation Loss')
ax[0].legend()

ax[1].plot(model2_history.history['acc'], label='Training Accuracy')
ax[1].plot(model2_history.history['val_acc'], label='Validation Accuracy')
ax[1].legend()

ax[2].plot(model2_history.history['mean_io_u_1'], label='Training Mean IoU')
ax[2].plot(model2_history.history['val_mean_io_u_1'], label='Validation Mean IoU')
ax[2].legend()
plt.savefig('./imgs/cc_seg_training_curves.png')

# %% [markdown]
# Simlarly to the previous model, the CCE model begins to overfit the training data at about 15 epochs. This model seems to score better on both accuracy and mean IoU. 

# %% [markdown]
# 


