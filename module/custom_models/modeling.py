# Uncomment for use in Google Colab
#!unzip processed.zip -d ./semantic_drone_dataset
#!pip install git+https://github.com/tensorflow/examples.git

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import cv2
import albumentations as A
from sklearn.model_selection import train_test_split
import pandas as pd

from tensorflow_examples.models.pix2pix import pix2pix

from IPython.display import clear_output
from glob import glob
import os

from data_loading.data_gen import DataGenerator
from data_loading.data_load import mask_decoding, mask_encoding
from custom_models.u_net_model import unet_model



transform = [
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Sharpen()
]


img_folder = '../semantic_drone_dataset/processed/images/'
mask_folder = '../semantic_drone_dataset/processed/label_images/'

img_files = np.sort(os.listdir(img_folder))
mask_files = np.sort(os.listdir(mask_folder))

# Train and Validation data genrator instances
train_generator = DataGenerator(img_folder, mask_folder, img_files, mask_files,batch_size=16, augment=transform, dim=[224, 224] ,shuffle=True, validation_split=0.2, subset='Train')
val_generator = DataGenerator(img_folder, mask_folder, img_files, mask_files,batch_size=16, augment=transform, dim=[224, 224] ,shuffle=True, validation_split=0.2, subset='Valid')




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

model2 = unet_model(output_channels=OUTPUT_CLASSES, name = 'cc_seg')
model2.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['acc',
                       tf.keras.metrics.MeanIoU(num_classes=24, sparse_y_pred=False, sparse_y_true=False)])


EPOCHS = 51

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('seg_model_cc.h5', 
                                                   save_best_only=True,
                                                   save_weights_only=True)

model2_history = model2.fit(train_generator, epochs=EPOCHS,
                          validation_data=val_generator,
                          callbacks=[DisplayCallback(), checkpoint_cb])


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



