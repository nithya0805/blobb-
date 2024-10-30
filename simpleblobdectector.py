import cv2
import numpy as np

image = cv2.imread('blobs.jpg')

params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 100
params.filterByCircularity = True
params.minCircularity = 0.9
params.filterByConvexity = True
params.minConvexity = 0.9
params.filterByInertia = True
params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(image)

blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(image,keypoints,blank,(0,0,255),cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

number_of_blobs =  len(keypoints)
text = "Number of circular blobs: "+str(len(keypoints))
cv2.putText(blobs,text,(20,550),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

cv2.imshow("Filtering Circular Blobs only",blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()


