# -*- coding: utf-8 -*-
"""
Created on Wed May 18 11:13:18 2022

@author: craig
"""

from __future__ import division
import  numpy as np
import cv2
import matplotlib.pyplot as plt
from pylab import *

class  Processing():
    
    def __init__(self, x, alpha, beta, eta):
        self.X = x
        height,width = x.shape
        self.height = height
        self.width = width
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        
    def neighbours(self, i, j):
        # Get the co-ordinate of all the neighbours surrounding the pixel.
        my_list = []
        # my_list.extend(([i-1,j-1], [i, j-1], [i-1, j-1]))
        # my_list.extend(([i+1,j], [i-1,j]))
        # my_list.extend(([i+1, j+1], [i, j+1], [i-1,j+1]))
        my_list.extend(([i-1,j+1],[i,j+1],[i+1,j+1]))
        my_list.extend(([i-1,j],[i+1,j]))
        my_list.extend(([i-1,j-1],[i,j-1],[i+1,j-1]))
        # Consider the end of the array
        # By iterating through each neighbour pixel.
        for l in range(len(my_list)):
            [m, n] = my_list[l]
            m = 0 if m<0 else m
            n = 0 if n<0 else n
            m = self.height-1 if m>=self.height else m
            n = self.width-1 if n>=self.width else n
            my_list[l] = [m,n]
        my_list[:] = [a for a in my_list if a != [i,j]]
        return my_list
    

    def probability(self, index_x, index_y):
        neighbours = self.neighbours(index_x,index_y)
        energy = self.alpha - self.beta*np.sum( self.X[x_val,y_val] for (x_val,y_val) in neighbours ) - self.eta*self.X[index_x,index_y]
        pos = np.exp(-energy)
        neg = np.exp(energy)
        ComponentA = float(pos)/(pos+neg)
        ComponentB = 1.0 - ComponentA
        return [ComponentB,ComponentA]  
    
    def selection(self, z):
        index_x, index_y = z
        prob_1, prob_2 = self.probability(index_x, index_y)
        self.X[i,j]=-1 if np.random.rand()<=prob_1 else 1 
                
# =============================================================================
# Import figure and pre-process  
# =============================================================================
plt.figure('1')
im = cv2.imread('y.png')
im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) 

plt.xlabel("pixels")
plt.ylabel("pixels")
plt.title('y.png')
# Tone image for simple  manipulation (Greyscale)
_,im_binary = cv2.threshold(im,127,1,cv2.THRESH_BINARY)
im_binary = (im_binary.astype(np.int) * 2) - 1

plt.figure('2')
im2 = cv2.imread('y.png')
im2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY) 

plt.xlabel("pixels")
plt.ylabel("pixels")
plt.title('x_gt.png')
# Tone image for simple  manipulation (Greyscale)
_2,im_binary2 = cv2.threshold(im2,127,1,cv2.THRESH_BINARY)
im_binary2 = (im_binary.astype(np.int) * 2) - 1
print(im_binary2[0:128])

imgplot = plt.imshow(im_binary, cmap = cm.Greys_r)
# =============================================================================
# Create second noisy figure
# =============================================================================
X = im_binary
height,width = X.shape
N = height*width
# =============================================================================
# Constants
# =============================================================================
alpha = 0.05#0.05
beta = 1
eta = 0.5#0.5
Size = 9

# =============================================================================
# Begin
# =============================================================================      
algorithm   = Processing(X, alpha, beta, eta)

prediction = np.zeros_like(X)
my_array = np.zeros_like(X)
for samp in range(Size):
  print(' current state = %d '%(samp*N))
  for i in range(0,height):
      for j in range(0,width):
            algorithm.selection([i,j])
            prediction += algorithm.X

im2 = (im2/255.0)
prediction = prediction.astype(float)
prediction = prediction/(N*Size)

prediction[prediction >= 0] = 1
prediction[prediction < 0] = 0
prediction = prediction.astype(np.int)
loss_function = 1/len(X)*np.sum(im2 - prediction)/N*Size
plt.figure()
plt.xlabel("pixels")
plt.ylabel("pixels")
plt.title('Denoised image')

imgplot = plt.imshow(prediction, cmap = cm.Greys_r)
plt.show()   
print("Here is the loss function: ", loss_function)

        
    