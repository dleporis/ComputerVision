# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 00:21:09 2020

@author: Damian Leporis
"""

import cv2
import numpy as np

import platform
print("running at: ",platform.system()) 


class Fruit:
    """In development, to improve the data organisation. This class contains feaures of detected fruit instance"""
    def __init__(self, contour, contour_area, height_width_ratio, circle_position, circle_r, circularity, label_color, classification):
        self.contour = contour
        self.contour_area = contour_area
        self.height_width_ratio = height_width_ratio
        self.circl_x = circle_position[0]
        self.circl_y = circle_position[1]
        self.circle_r = circle_r
        self.circularity = circularity
        self.label_color = label_color
        self.classification
        

def resize(source, scale_percent):
    width = int(source.shape[1] * scale_percent / 100)
    height = int(source.shape[0] * scale_percent / 100)
    dsize = (width, height)
    # resize image
    resized = cv2.resize(source, dsize, interpolation = cv2.INTER_CUBIC)
    return resized

def segmentation(image):
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray,9)
    # apply threshold to make veggies white and background white
    _, thresholded = cv2.threshold(blur, 193, 255, cv2.THRESH_BINARY_INV)
    # if necessary, apply morphology to reduce noise and close holes
    output = morphology(thresholded)
    return output
    
def morphology(image):    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    open_img = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    close_img = cv2.morphologyEx(open_img, cv2.MORPH_CLOSE, kernel)
    return close_img

def featureExtraction(inputImg, originalImg):
    features = {} #dictionary of features
    #find contours from the thresholded image
    contours, hierarchy = cv2.findContours(inputImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)# can also use cv2.CHAIN_APPROX_NONE
   #iterate through found contours
    for contour in range(len(contours)):
        # only shapes with specific aea will be considered
        if 10 ** 4 < cv2.contourArea(contours[contour]) < 10 ** 7:
            #find the minimum area rectangle which touches the object contours from oustide
            minBox = cv2.minAreaRect(contours[contour])
            #find ratio between the "width" and "length" of the fruit
            ratio = min(minBox[1]) / max(minBox[1])
            
            # The smaller the ratio, the longer the veggie
            if ratio < .25:
                features.setdefault(contour, "cucumber")
            elif ratio < .5:
                features.setdefault(contour, "banana")
                
            # here is more round fruit
            
            else:
                # we can use minimum area circle drawn around the object boundries
                radius = max(minBox[1])/2
                circle_area = np.pi * radius * radius
                shape_area = cv2.contourArea(contours[contour])
                #circularity is a ratio of the circle ant the actual area of fruit
                #perfectly circular fruit would have circularity == 1
                circularity = shape_area /circle_area
                #categorise fruit based on the circularity
                if circularity > .9:
                    features.setdefault(contour, "orange")
                elif circularity > .7:
                    features.setdefault(contour, "apple")
                else:
                    features.setdefault(contour, "pear")
                print("Circularity index of ", features.get(contour)," is ", circularity)
            print("Width to length ratio of", features.get(contour)," is ", ratio,"\n")
                
    print("features: ", features)
    return features, contours
    
def showLabels(image, featureDict, contours):
    """This function changes the color value accurding to the feature classification.
    Then it draws contours of the given fruit with this color"""
    #iterate through feature dictionary items
    for entry in featureDict:
        print(featureDict.get(entry))
        if featureDict.get(entry) == "carrot":
            col  = (49,96,226)  # 8bit  [0-255] values representing blue; green; red (BGR) colors
        elif featureDict.get(entry) == "cucumber":
            col = (0, 255, 0)
        elif featureDict.get(entry) == "banana":
            col = (0, 215,255)
        elif featureDict.get(entry) == "pear":
            col = (0,255, 126)
        elif featureDict.get(entry) == "apple":
            col = ( 0, 0,255)
        elif featureDict.get(entry) == "orange":
            col = ( 200,165,0)
        elif featureDict.get(entry) == "pepper":
            col = (239,69,19)
        else:
            # if a entry in the dictionary is not correctly identified,
            # the function returns false without ever printing the contours in the image
            raise Exception("printContours: A entry in the dictionary is not correctly identified")
        # Draws the contour with index "entry" in the list "contours"
        cv2.drawContours(image, contours, entry, col, 5) #5 is the line thickness


def showImages(array):    
    cv2.startWindowThread()
    
    for i in range(0, len(array[0])):
        #cv2.imshow(caption, image)
        cv2.imshow(array[0][i], array[1][i])
        print("showed img ", i)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return

def main():
    showArr = [] #list that storesimages that will be displayed
    #add sublists
    showArr.append([]) #this will store image window caption
    showArr.append([]) #this will store images themselves
    
    # To see all steps: True; To hide the steps: False
    showSteps = False
    
    #image aquisition has been done by a phone camera, Samsung Galaxy S8
    img = cv2.imread('fruit.jpg', cv2.IMREAD_UNCHANGED)
    if showSteps == True:
        showArr[0].append("raw image")
        showArr[1].append(img)
    
    #resize picture
    imgR = resize(img,40) #2nd argument is percentage of new image size to the original one
    imgR2 = imgR #create a copy, so it doesn't get affected by classification
    if showSteps == True:
        showArr[0].append("resized image")
        showArr[1].append(imgR2)
    
    #perform segmentation steps: grayscale, threshold    
    segmented = segmentation(imgR)
    if showSteps == True:
        showArr[0].append("segmented image")
        showArr[1].append(segmented)
    
    #extact features from the segments
    feat, cont = featureExtraction(segmented, imgR)
    
    #draw coloured contours and labels according to the determined class of object
    showLabels(imgR, feat, cont)
    
    showArr[0].append("classified fruit")
    showArr[1].append(imgR)
    
    #show everything
    showImages(showArr)
    
    return 0 
    #end main

#make sure that file is running directly, not as imported
if __name__ == '__main__':
    
    #call main function
    main()
