import sys

import tensorflow as tf
import keras
from keras import datasets, layers, models
from keras.layers import Dropout
import matplotlib.pyplot as plt

class MyModel:

    def __init__(self):
        
        self.Conv2D_1 = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))
        self.Conv2D_2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.Conv2D_3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.Conv2D_4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.Conv2D_5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.Conv2D_6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.Conv2D_7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pooling_1 = layers.MaxPooling2D((2, 2))
        self.pooling_2 = layers.MaxPooling2D((2, 2))
        self.pooling_3 = layers.MaxPooling2D((2, 2))
        self.pooling_4 = layers.MaxPooling2D((2, 2))
        self.pooling_5 = layers.MaxPooling2D((2, 2))
        self.pooling_6 = layers.MaxPooling2D((2, 2))
        self.dropout_1 = Dropout(0.25)
        self.dropout_2 = Dropout(0.25)
        self.dropout_3 = Dropout(0.25)
        self.flatten = layers.Flatten()
        self.dense_1 = layers.Dense(128, activation='relu')
        self.dense_2 = layers.Dense(10, activation='softmax')


    def model_create(self):
        model = models.Sequential()
        model.add(self.Conv2D_1)
        model.add(self.pooling_1)
        model.add(self.Conv2D_2)
        model.add(self.pooling_2)
        model.add(self.Conv2D_3)
        model.add(self.pooling_3)
        model.add(self.Conv2D_4)
        model.add(self.pooling_4)
        model.add(self.Conv2D_5)
        model.add(self.dropout_1)
        model.add(self.flatten)
        model.add(self.dropout_2)
        model.add(self.dense_1)
        model.add(self.dense_2)

        model.summary()

        model.compile(optimizer='adam', 
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
                      metrics=['accuracy'])
        
        return model
    

def main(preview=True):

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255, test_images / 255

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    if preview == True:     
        plt.figure(figsize=(10,10))
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i])
            plt.xlabel(class_names[train_labels[i][0]])
        plt.show()
     
    model = MyModel().model_create()

    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print(test_acc)
    print(test_loss)

if __name__ == '__main__':
    main(preview=True)

