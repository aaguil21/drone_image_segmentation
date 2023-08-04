import tensorflow as tf
from tensorflow import keras

class DataGenerator(tf.keras.utils.Sequence):
    """
    Custom generator for use with image segmentation. Initiating the generator requires the folders and file lists for images
    and masks. 
    """
    
    def __init__(self, img_folder,mask_folder, img_path, mask_path, batch_size=16, dim=[224,224], augment=None, shuffle=True, 
                validation_split = 0.0, subset = None):
        'Initialization'
        self.dim = dim        
        # Create train and validation splits in the data generation            
        split = int(np.floor(len(img_path) * (1-validation_split)))
        if subset == 'Train':
            self.img_path = img_path[:split]
            self.mask_path = mask_path[:split]
        elif subset == 'Valid':
            self.img_path = img_path[split:]
            self.mask_path = mask_path[split:]
        else:
            self.img_path = img_path
            self.mask_path = mask_path
            
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Generate the number of batches created
        """
        return int(np.floor(len(self.img_path) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [k for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        
        # Add augmented images to dataset from augment input
        if self.augment is None:
            return X, y
        else:
            im, mask = [], []
            for x, y in zip(X, y):
                im.append(x)
                mask.append(to_categorical(y, num_classes = 24))
                for augment in self.augment:
                  augmented = augment(image=x, mask=y)
                  im.append(augmented['image'])
                  mask.append(to_categorical(augmented['mask'], num_classes = 24))
            return np.array(im), np.array(mask)     
            
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.img_path))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # Initialization
        batch_imgs = []
        batch_labels = []

        # Generate data
        for i in list_IDs_temp:

            img_file = tf.io.read_file(self.img_folder + self.img_path[i])
            img = tf.image.decode_png(img_file, channels=3, dtype=tf.uint8)
            img = tf.image.resize(img, self.dim, method='nearest')
            img = img_to_array(img)/255.
            batch_imgs.append(img)

            label_file = tf.io.read_file(self.mask_folder + self.mask_path[i])
            label = tf.image.decode_png(label_file, channels=1, dtype=tf.uint8)
            label = tf.image.resize(label, self.dim, method='nearest')
            if self.augment is None:
              label = to_categorical(label , num_classes = 24)
            batch_labels.append(label)
            
        return np.array(batch_imgs) ,np.array(batch_labels)