import cv2
import numpy as np

image = cv2.imread("class 6/blob.jpg")

para=cv2.SimpleBlobDetector_Params()

para.filterByArea=True
para.minArea=100
para.filterByCircularity=True
para.minCircularity=0.8
para.filterByConvexity = True
para.minConvexity=0.9
para.filterByInertia=True
para.minInertiaRatio=0.01

detector=cv2.SimpleBlobDetector_create(para)

keypoint=detector.detect(image)

blank=np.zeros((1,1))
blobs=cv2.drawKeypoints(image,keypoint,blank,(0,0,255),cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

numblobs=len(keypoint)
text="Number of circular blobs: "+str(len(keypoint))
cv2.putText(blobs,text,(30,500),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)

cv2.imshow("Filtering Circular blobs only" ,blobs)
cv2.waitKey(0)

cv2.destroyAllWindows()



