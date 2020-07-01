 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as nm
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm
from pandas.plotting import scatter_matrix
import time
import matplotlib.pyplot as plt

data = pd.read_csv('product_images.csv')

def dataCleaning():
  #making targets to extract features with that label
   targetS = data['label'] == 1
   targetA = data['label'] == 0
   
   
   #Using targets to extract features form the dataset
   sneakers = data['label'][targetS]
   
   ankleBoots = data['label'][targetA]
   
   features = data.drop(columns=['label'])
   
  
   #getting the length of sneakers and anke
   noOfSneakers = len(sneakers)
   noOfAnkleBoots = len(ankleBoots)
  
  
   #making set for first training run
   dataset = nm.array(features.sort_index())
   target = sneakers.append(ankleBoots) 
  
   
   return dataset, target , noOfSneakers , noOfAnkleBoots


def showAnkle(dataset):
   print("Image of AnkleBoot")
   image1 = dataset[0,:]
   plt.imshow(image1.reshape(28,28))
   plt.show()
   
def showSneaker(dataset):
   print("Image of sneaker")
   image2 = dataset[3,:]
   plt.imshow(image2.reshape(28,28))
   plt.show()
   

def perceptron(dataset, target, sneakerLength,anklebootsLength):
   
   print("The number of Sneakers in the dataset are : \n",sneakerLength)
   print("The number of AnkleBoots in the dataset are : \n",anklebootsLength)

   
   
   
   
   score = []
   kf = KFold(10,True,1)
   
   for train_index , test_index in kf.split(dataset):
   
       clf = linear_model.Perceptron()
       start = time.time() 
       clf.fit(dataset[train_index],target[train_index])
       print("Time Taken for Perceptron Training: %s" % (time.time() - start))
       
       start = time.time()    
       prediction = clf.predict(dataset[test_index])
       print("Time Taken for Perceptron Prediction: %s" % (time.time() - start))
   
       cMatrix = confusion_matrix(target[test_index],prediction)
       accuracy = accuracy_score(target[test_index],prediction)
       
       score.append(accuracy)
       print("----------Perceptron-------------")
       print("Confusion Matrix :\n",cMatrix)
       print("---------------------------------")
       print("Accuracy Score :\n",accuracy)
       print("---------------------------------")
   print("Min Accuracy", nm.min(score))
   print("---------------------------------")
   print("Mean Accuracy",nm.mean(score))
   print("---------------------------------")
   print("Max Accuracy", nm.max(score))
   print("---------------------------------")
   plt.plot(score)
   plt.show()
   
   
def SVMLinear(dataset,labels):
   
   dataset = dataset[0:7000]
   
   
   kf = KFold(10,True,1)
   score = []
   
   for train_index , test_index in kf.split(dataset):
       clf = svm.SVC(kernel="linear")
       
       start = time.time() 
       clf.fit(dataset[train_index],labels[train_index])
       print("Time Taken for Support Vector Machine Training: %s" % (time.time() - start))
      
       start = time.time() 
       prediction = clf.predict(dataset[test_index])
       print("Time Taken for Support Vector Machine Prediction : %s" % (time.time() - start))
       
       c = confusion_matrix(labels[test_index],prediction)
       a = accuracy_score(labels[test_index],prediction)
       score.append(a)
      
       print("----------Support Vector Machine Linear Kernel-------------")
       print("Confusion Matrix :\n",c)
       print("-----------------------------------------------------------")
       print("Accuracy Score :\n",a)
   print("---------------------------------------------------------------")
   print("Min Accuracy", nm.min(score))
   print("---------------------------------------------------------------")
   print("Mean Accuracy",nm.mean(score))
   print("---------------------------------------------------------------")
   print("Max Accuracy", nm.max(score))
   print("---------------------------------------------------------------")
   plt.plot(score)
   plt.show()
   
def SVMRadial(dataset,labels):
   
   dataset = dataset[0:7000]
   gammas = [0000.01,000.01,00.01,0.01,1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]
   
   kf = KFold(6,True,1)
   score = []
   
   for train_index , test_index in kf.split(dataset):
       clf = svm.SVC(gamma=gammas[13],kernel="rbf")
       
       start = time.time() 
       clf.fit(dataset[train_index],labels[train_index])
       print("Time Taken for Support Vector Machine Training: %s" % (time.time() - start))
      
       start = time.time() 
       prediction = clf.predict(dataset[test_index])
       print("Time Taken for Support Vector Machine Prediction : %s" % (time.time() - start))
       
       c = confusion_matrix(labels[test_index],prediction)
       a = accuracy_score(labels[test_index],prediction)
       score.append(a)
      
       print("----------Support Vector Machine Radial Kernel-------------")
       print("Confusion Matrix :\n",c)
       print("-----------------------------------------------------------")
       print("Accuracy Score :\n",a)
   print("---------------------------------------------------------------")
   print("Min Accuracy", nm.min(score))
   print("---------------------------------------------------------------")
   print("Mean Accuracy",nm.mean(score))
   print("---------------------------------------------------------------")
   print("Max Accuracy", nm.max(score))
   print("---------------------------------------------------------------")
   plt.plot(score)
   plt.show()   
   
  
   
       
   
   
   
   
   
  
   




def main():
    dataset,labels ,lensneakers,lenankleboots = dataCleaning()
    showAnkle(dataset)
    showSneaker(dataset)
    perceptron(dataset,labels,lensneakers,lenankleboots)
    SVMLinear(dataset,labels)
    SVMRadial(dataset,labels)
    
    
    
    
    
main()