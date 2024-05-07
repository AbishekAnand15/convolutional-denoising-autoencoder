# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
In this experiment, we use an autoencoder to process handwritten digit images from the MNIST dataset. The autoencoder learns to encode and decode the images, reducing noise through layers like MaxPooling and convolutional. Then, we repurpose the encoded data to build a convolutional neural network for classifying digits into numerical values from 0 to 9. The goal is to create an accurate classifier for handwritten digits removing noise.
### Dataset
![image](https://github.com/Jenishajustin/convolutional-denoising-autoencoder/assets/119405070/e9551a34-7cfa-4058-95af-25a04c61cd89)

## Convolution Autoencoder Network Model
![Screenshot 2024-04-24 203449](https://github.com/Jenishajustin/convolutional-denoising-autoencoder/assets/119405070/709637a2-f1d9-437a-a962-7070e8e9b333)

## DESIGN STEPS

### STEP 1:
Import the necessary libraries and load the mnist dataset without label column (y).
### STEP 2:
Scale the input (gray scale) images between 0 to 1.
### STEP 3:
Add noise to the input image and scale the noised image between 0 and 1.
### STEP 4:
Build the Neural Network model with
<ol>
  <li>Encoder</li>
<ul>
  <li>Convolutional layer</li>
  <li>Max Pooling (downsampling) layer</li>
</ul>
  <li>Decoder</li>
  <ul>
    <li>Convolutional layer</li>
    <li>Upsampling layer</li>
  </ul>
</ol>

### STEP 5:
Compile and fit the model.
### STEP 6:
Plot the predictions.

## PROGRAM
```
Developed by : Abishek Xavier A
Reg.No       : 212222230004
```
```python
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train,_),(x_test,_)=mnist.load_data()
x_train.shape

x_train_scaled=x_train.astype('float32')/255.
x_test_scaled=x_test.astype('float32')/255.
x_train_scaled=np.reshape(x_train_scaled,(len(x_train_scaled),28,28,1))
x_test_scaled=np.reshape(x_test_scaled,(len(x_test_scaled),28,28,1))

noise_factor=0.5
x_train_noisy=x_train_scaled+noise_factor*np.random.normal(loc=0.0,scale=1.0,size=x_train_scaled.shape)
x_test_noisy=x_test_scaled+noise_factor*np.random.normal(loc=0.0,scale=1.0,size=x_test_scaled.shape)

x_train_noisy=np.clip(x_train_noisy,0.,1.)
x_test_noisy=np.clip(x_test_noisy,0.,1.)

n=10
plt.figure(figsize=(20,2))
for i in range(1,n+1):
  ax=plt.subplot(1,n,i)
  plt.imshow(x_test_noisy[i].reshape(28,28))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()

input_img = keras.Input(shape=(28, 28, 1))

x=layers.Conv2D(16,(3,3),activation='relu',padding='same')(input_img)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(6,(7,7),activation='relu',padding='same')(x)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(8,(3,3),activation='relu',padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

x=layers.Conv2D(8,(3,3),activation='relu',padding='same')(encoded)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(6,(7,7),activation='relu',padding='same')(x)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(16,(3,3),activation='relu')(x)
x=layers.UpSampling2D((2,2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train_noisy, x_train_scaled,
                epochs=2,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))

print("Abishek Xavier A\n212222230004")
metrics = pd.DataFrame(autoencoder.history.history)
metrics[['loss','val_loss']].plot()

decoded_imgs = autoencoder.predict(x_test_noisy)

print("Abishek Xavier A\n212222230004")
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display noisy
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    

    # Display reconstruction
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot & Original vs Noisy Vs Reconstructed Image:-
![Screenshot 2024-05-07 110909](https://github.com/AbishekAnand15/convolutional-denoising-autoencoder/assets/118706942/c712bc71-d633-43f0-a2a8-d1eda828569c)





## RESULT
Thus, the convolutional autoencoder for image denoising application has been successfully developed.
