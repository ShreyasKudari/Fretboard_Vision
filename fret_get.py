from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()
capture = cv.VideoCapture(0)
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    #might not need backsub but lets have it for now
    fgMask = backSub.apply(frame)
    kernel = np.ones((5,5),np.uint8)
    fgMask = cv.dilate(fgMask,kernel,iterations=1)
    masked = cv.bitwise_and(frame, frame, mask = fgMask)
    gray = cv.cvtColor(masked,cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray,50,75,apertureSize = 3)
    fret_mask = np.zeros(frame.shape, np.uint8) #edges or frame
    lines = cv.HoughLinesP(edges,1,np.pi/180,200,minLineLength=200,maxLineGap=8)
    
    if lines is not None and len(lines)>=4:
        median_line = lines[int(len(lines)/2)] #median is always contained in the majority set
        x1,y1,x2,y2 = median_line[0]
        if x2-x1 == 0:
            med_grad = 10000
        else: 
            med_grad = (y2-y1)/(x2-x1)

        newlines = []
        min_x, max_x = -1,-1
        for line in lines:
            x1,y1,x2,y2 = line[0]
            if x2-x1 == 0:
                grad = 10000
            else: 
                grad = (y2-y1)/(x2-x1)

            if np.absolute(grad-med_grad)<0.1: #get the lines close to majority gradient
                #get bounds in x axis
                if min_x==-1:
                    if x1<x2:
                        min_x = x1
                        max_x = x2
                    else:
                        min_x = x2
                        max_x = x1
                else:
                    if x1<x2 and x1<min_x:
                        min_x = x1
                    if x2<x1 and x2<min_x:
                        min_x = x2
                    if x1>x2 and x1>max_x:
                        max_x = x1
                    if x2>x1 and x2>max_x:
                        max_x = x2
                newlines.append(line)
                
        #extend all lines along their gradients to the bounds
        #extended_lines = []
        for line in newlines:
            x1,y1,x2,y2 = line[0]
            if x1<x2 and x1>min_x:
                if x2-x1 == 0:
                    grad = 10000
                else: 
                    grad = (y2-y1)/(x2-x1)
                y1 = y1-grad*(x1-min_x)
                x1 = min_x
                if x2<max_x:
                    y2 = y2+grad*(max_x-x2)
                    x2 = max_x
            if x1>x2 and x2>min_x:
                if x2-x1 == 0:
                    grad = 10000
                else: 
                    grad = (y2-y1)/(x2-x1)
                y2 = y2-grad*(x2-min_x)
                x2 = min_x
                if x1<max_x:
                    y1 = y1+grad*(max_x-x1)
                    x1 = max_x
            line[0][0] = x1
            line[0][1] = y1
            line[0][2] = x2
            line[0][3] = y2

        for line in newlines:
            x1,y1,x2,y2 = line[0]
            cv.line(masked,(x1,y1),(x2,y2),(0,255,0),2)
        
        #create bounding box
        
        min_y_left, max_y_left = -1,-1
        min_y_right, max_y_right = -1,-1
        for line in newlines:
            x1,y1,x2,y2 = line[0]
            if x1==min_x:
                y = y1
            else: 
                y = y2

            if min_y_left==-1:
                min_y_left = y
                max_y_left = y
            elif y<min_y_left:
                min_y_left = y
            elif y>max_y_left:
                max_y_left = y

        for line in newlines:
            x1,y1,x2,y2 = line[0]
            if x1==max_x:
                y = y1
            else: 
                y = y2

            if min_y_right==-1:
                min_y_right = y
                max_y_right = y
            elif y<min_y_right:
                min_y_right = y
            elif y>max_y_right:
                max_y_right = y

        
        #add padding
        
        
        roi_corners = np.array([[(min_x-2, min_y_left-2),(max_x+2, min_y_right-2),(max_x+2,max_y_right+2),(min_x-2,max_y_left+2)]], np.int32)
        cv.fillPoly(fret_mask, roi_corners, (255,255,255))
        theta = np.arctan(med_grad)*180/np.pi
        #apply mask on 'frame!!!' -> output of canny edge detector applied on foreground masked
        fret_view = cv.bitwise_and(frame,fret_mask)
        # print(fret_view.shape)
        '''
            Here I have two options: rotate the fret_view to zero gradient or keep as it is...rotating to zero
            is done with the help of the median gradient. However, it is quite noisy..therefore trying without rotating.
            This should not be a big issue since guitars necks while playing almost always are close to horizontal (non-classical)
        '''
        # rows, cols = fret_view.shape[:2]
        # M = cv.getRotationMatrix2D((cols/2,rows/2),theta, 1)
        # fret_view = cv.warpAffine(fret_view,M,(cols,rows)) #rotated to prepare for sobel filtering

        grad_x = cv.Sobel(fret_view, cv.CV_16S, 1, 0, ksize = 3) #also try out Scharr()
        abs_grad_x = cv.convertScaleAbs(grad_x)
        _,thresh1 = cv.threshold(abs_grad_x,220,255,cv.THRESH_BINARY)
        filtered = cv.medianBlur(thresh1,5)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
        filtered = cv.morphologyEx(filtered, cv.MORPH_CLOSE, kernel, iterations = 1)
        filtered = cv.dilate(filtered, kernel, iterations=3)
        grayed = cv.cvtColor(filtered, cv.COLOR_RGBA2GRAY)
        edges = cv.Canny(grayed,100,150,apertureSize = 3)
        contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        
        rects = []
  
        #get boxes that can be fret enclosings
        for cnt in contours:
            if cv.contourArea(cnt)>200:
                rect = cv.minAreaRect(cnt)
                box = cv.boxPoints(rect)
                box = np.int0(box)
                rects.append(box)
            

        
        #place pointers on all fret string intersections
        frets = []
        for box in rects:
            x1,y1 = box[2]
            x2,y2 = box[3]
            frets.append((x1,y1,x2,y2))
            margin = 1
            cv.line(frame,(x1,y1-margin),(x2,y2+margin),(0,0,255),2)

        # cv.drawContours(frame, rects, -1, (0,255,0), 1)
        frets.sort(key = lambda x: x[1], reverse = True)

        for fret in frets:
            x1,y1, x2,y2 = fret
            dist = ((y2-y1)**2+(x2-x1)**2)**0.5
            step = dist/6
            error = 1
            for i in range(6):
                if x1==x2:
                    cv.circle(frame, (x1,int(y1+step*i)+error), radius=1, color=(0, 255, 0), thickness=-1)
                else:
                    slope = (y2-y1)/(x2-x1)
                    angle = np.arctan(slope)
                    xpt = int(x1+(step*i)*np.cos(angle))
                    ypt = int(y1+(step*i)*np.sin(angle))
                    cv.circle(frame, (xpt+error,ypt+error), radius=3, color=(0, 255, 0), thickness=-1)

                


        cv.imshow('frets', frame)

      

    cv.imshow('Frame', masked)
    # cv.imshow('ROI', fret_mask)
    # cv.imshow('cannyedges', edges)
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break