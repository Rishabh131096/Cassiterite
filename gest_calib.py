import cv2
import math

#-------Declare global variables
fingers = []          #store coordinates of each finger
fin=[0,0,0,0,0]       #check which fingers are raised
calibrated = False    #enter and exit calibration mode

#-------Gesture Command 0-tumb, 1-index finger, ...
table={0:"fist",1:"one",2:"two",3:"three",4:"four",5:"five"}


########Function to predict the gesture from table
def predictGesture(fin):
	sum = 0
	for i in range(0,5):
		sum=sum*2+fin[i]
	
	return table.get(sum,'DEFAULT') #if value doesn't exist return 'DEFAULT'


########Function to guess which finger is raised
def guessFinger(x,y):
	global fingers
	THRESHOLD = 20
	
	for i in range(0,5):
		dist = math.sqrt((x-fingers[i][0])*(x-fingers[i][0]) + (y-fingers[i][1])*(y-fingers[i][1]))
		if(dist < THRESHOLD):
			return i
	
	return -1

	
########Function to enter Calibration Mode
def calib(img2):
	img1 = img2[100:400,100:400]
	img1 = cv2.GaussianBlur(img1,(5,5),0)
	img=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
	
	global fingers
	global calibrated
	
	fingers = []

####################thresholding
	__,thresh = cv2.threshold(img.copy(),60,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	cv2.imshow("HAND",thresh)

	
####################finding contours
	_,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	if(contours is None):
		return img2

		
####################largest contour
	cnt=max(contours,key=lambda x: cv2.contourArea(x))
	if(cnt is None):
		return img2

	cv2.drawContours(img1, [cnt] , 0, (0,255,0), 3)

	
####################convex hull
	hull = cv2.convexHull(cnt)
	cv2.drawContours(img1,[hull],0,(0,0,255),3)

	
###################defects
	hull = cv2.convexHull(cnt, returnPoints=False)
	defects = cv2.convexityDefects(cnt, hull)
	if(defects is None):
		return img2

	defectpoints=[]
	count=0

	maxi=-1
	maxInd = -1
	
	THRESHOLD = 20
	tDist = 25
	maxdis=THRESHOLD
	perpDist=0
	for i in range(defects.shape[0]):
		s,e,f,d = defects[i,0]
		start = tuple(cnt[s][0])
		end = tuple(cnt[e][0])
		far = tuple(cnt[f][0])
    
	# find length of all sides of triangle
		a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
		b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
		c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

    # apply cosine rule here
		angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
		dis1=math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
		dis2=math.sqrt((start[0] - far[0])**2 + (start[1] - far[1])**2)
    
	# ignore angles > 90 and highlight rest with red dots
		perpDist = (abs( (end[1]-start[1])*far[0] - (end[0]-start[0])*far[1] + end[0]*start[1] - end[1]*start[0]))/a
		
		if(perpDist>tDist):
		
			if(dis1>maxdis):
				maxdis=dis1
				maxi=i
    
			if(dis2>maxdis):
				maxdis=dis2
				maxi=i
	
			if angle <= 100 and (dis1>50 or dis2>50):
				cv2.circle(img1, far, 4, [255,0,255], -1)
				defectpoints.append(i)
				count=count+1


	if(len(defectpoints)==4):
		# INSERTING COORDINATES OF FINGERS
		s,e,f,d=defects[defectpoints[0],0]
		fingers.append([cnt[s][0][0],cnt[s][0][1]])
		cv2.circle(img1,(cnt[s][0][0],cnt[s][0][1]),4,[0,255,255],-1)
		
		for j in range(0,len(defectpoints)-1):
			s,e,f,d=defects[defectpoints[j],0]
			s2,e2,f2,d2=defects[defectpoints[j+1],0]
	
			pointx=int((cnt[e][0][0]+cnt[s2][0][0])/2)
			pointy=int((cnt[e][0][1]+cnt[s2][0][1])/2)
			fingers.append([pointx,pointy])
			cv2.circle(img1, (pointx,pointy), 4, [0,255,255], -1)
		
		s,e,f,d=defects[defectpoints[len(defectpoints)-1],0]
		fingers.append([cnt[e][0][0],cnt[e][0][1]])
		cv2.circle(img1,(cnt[e][0][0],cnt[e][0][1]),4,[0,255,255],-1)
		
		#########CALIBRATED
		calibrated = True
			
	img2[100:400,100:400] = img1
	cv2.rectangle(img2,(100,100),(400,400),(255,0,0))
	return img2,"Calibration"

	
##########Function to process image to find fingers, then predict gesture
def predict(img2):
	global fin
	fin=[0,0,0,0,0]
	img1 = img2[100:400,100:400]
	img1 = cv2.GaussianBlur(img1,(5,5),0)
	img=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)


####################thresholding
	__,thresh = cv2.threshold(img.copy(),60,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	cv2.imshow("HAND",thresh)

	
####################finding contours
	_,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	if(contours is None):
		return img2

		
####################largest contour
	cnt=max(contours,key=lambda x: cv2.contourArea(x))
	if(cnt is None):
		return img2

	cv2.drawContours(img1, [cnt] , 0, (0,255,0), 3)

	
####################convex hull
	hull = cv2.convexHull(cnt)
	cv2.drawContours(img1,[hull],0,(0,0,255),3)

################### defects
	hull = cv2.convexHull(cnt, returnPoints=False)
	defects = cv2.convexityDefects(cnt, hull)
	if(defects is None):
		return img2

	defectpoints=[]
	count=0

	maxi=-1
	maxInd = -1
	
	THRESHOLD = 20
	tDist = 25
	maxdis=THRESHOLD
	perpDist=0
	for i in range(defects.shape[0]):
		s,e,f,d = defects[i,0]
		start = tuple(cnt[s][0])
		end = tuple(cnt[e][0])
		far = tuple(cnt[f][0])
    # find length of all sides of triangle
		a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
		b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
		c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

    # apply cosine rule here
		angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
		dis1=math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
		dis2=math.sqrt((start[0] - far[0])**2 + (start[1] - far[1])**2)
    # ignore angles > 90 and highlight rest with red dots
		perpDist = (abs( (end[1]-start[1])*far[0] - (end[0]-start[0])*far[1] + end[0]*start[1] - end[1]*start[0]))/a
		
		if(perpDist>tDist):
			
			if(dis1>maxdis):
				maxdis=dis1
				maxi=i
    
			if(dis2>maxdis):
				maxdis=dis2
				maxi=i
	
			if angle <= 100 and (dis1>50 or dis2>50):
				cv2.circle(img1, far, 4, [255,0,255], -1)
				defectpoints.append(i)
				count=count+1

	for j in range(0,len(defectpoints)-1):
		s,e,f,d=defects[defectpoints[j],0]
		s2,e2,f2,d2=defects[defectpoints[j+1],0]
	
		pointx=int((cnt[e][0][0]+cnt[s2][0][0])/2)
		pointy=int((cnt[e][0][1]+cnt[s2][0][1])/2)
		cv2.circle(img1, (pointx,pointy), 4, [0,255,255], -1)
		if(guessFinger(pointx,pointy)!=-1):
			fin[guessFinger(pointx,pointy)]=1
		
	if(count!=0):
		s,e,f,d=defects[defectpoints[0],0]
		cv2.circle(img1,(cnt[s][0][0],cnt[s][0][1]),4,[0,255,255],-1)	
		if(guessFinger(cnt[s][0][0],cnt[s][0][1])!=-1):
			fin[guessFinger(cnt[s][0][0],cnt[s][0][1])]=1

	if(count!=0):
		s,e,f,d=defects[defectpoints[len(defectpoints)-1],0]
		cv2.circle(img1,(cnt[e][0][0],cnt[e][0][1]),4,[0,255,255],-1)	
		if(guessFinger(cnt[e][0][0],cnt[e][0][1])!=-1):
			fin[guessFinger(cnt[e][0][0],cnt[e][0][1])]=1	
	if(count==0):
		
		if(maxdis > THRESHOLD):
			s,e,f,d = defects[maxi,0]
			start = tuple(cnt[s][0])
			end = tuple(cnt[e][0])
			far = tuple(cnt[f][0])
		
			if(end[1]<start[1]):
				cv2.circle(img1, (end[0],end[1]), 4, [0,255,255], -1)
				if(guessFinger(end[0],end[1])!=-1):
					fin[guessFinger(end[0],end[1])]=1
			else:
				cv2.circle(img1, (start[0],start[1]), 4, [0,255,255], -1)
				if(guessFinger(start[0],start[1])!=-1):	
					fin[guessFinger(start[0],start[1])]=1
	
		
	img2[100:400,100:400] = img1
	cv2.rectangle(img2,(100,100),(400,400),(255,0,0))
	return img2,fin

	
#----------Start capturing from webcam
cap = cv2.VideoCapture(0)


End_of_Video = False
while(1):
	
	ret, img = cap.read()
	if ret==False:
		End_of_Video = True
		break 
	if(not calibrated):
		predicted_img,text = calib(img)
		predicted_img = cv2.flip(predicted_img,1)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(predicted_img, text, (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

	else:
		predicted_img,fin = predict(img)
		gesture = predictGesture(fin)
		predicted_img = cv2.flip(predicted_img,1)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(predicted_img, gesture, (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
					
	cv2.imshow("VIDEO",predicted_img)
	
	
	k = cv2.waitKey(1)
	if k == 27:
		break
	if (k == ord('c')):
		calibrated = False

cv2.destroyAllWindows()



