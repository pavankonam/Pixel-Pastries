import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from scipy.linalg import sqrtm
from tensorflow.keras.losses import BinaryCrossentropy

cross_entropy = BinaryCrossentropy(from_logits=True) #Loss Function

def discriminator_loss(real_output, fake_output):
    #Calculating the Discriminator Loss
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    #Calculating the Generator Loss
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def calculate_fid(real_images, generated_images):
    #Calculating the Frenchet Inception Distance(FID) score (lower the better)
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(75, 75, 3))
    def preprocess(imgs):
        imgs = tf.image.resize(imgs, (75, 75)) #Resizing to match InceptionV3
        return preprocess_input(imgs)
    act1 = model.predict(preprocess(real_images))
    act2 = model.predict(preprocess(generated_images))
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0*covmean)
    return fid