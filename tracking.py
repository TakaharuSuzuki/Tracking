#-------------------------------------------------------------------------------
# Name:        tracking.py
# Purpose:
#
# Author:      Takaharu Suzuki
#
# Created:     26/03/2015
# Copyright:   (c) Takaharu Suzuki 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

# impport necesarry modules
import numpy as np
import cv2
import cv2.cv as cv
import Image
import ImageOps

# mouse click function
(cx, cy) = (0, 0)
def onMouse(event, x, y, flags, param):
    global cx,cy,click_r
    if event == cv2.EVENT_MOUSEMOVE:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        (cx, cy) = (x, y)
        return
    if event == cv2.EVENT_RBUTTONDOWN:
        return

# SURF key point function
def surfKeyPoint(input):
    surf = cv2.SURF(5000)
    kp, des = surf.detectAndCompute(input,None)
    output = cv2.drawKeypoints(input,kp,None,(0,0,255),4)
    return output

# SIFT key point function
def siftKeyPoint(input):
    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(input,None)
    output = cv2.drawKeypoints(input,kp,None,(0,0,255),4)
    return output

# Shi and Tomasi key point function
def ShiTomasiKeyPoint(input):
    corners = cv2.goodFeaturesToTrack(input,25,0.01,10)
    corners = np.int0(corners)
    output = input
    for i in corners:
        x,y = i.ravel()
        cv2.circle(output,(x,y),10,(0,0,255),2)
    return output

# FAST key point function
def fastKeyPoint(input):
    fast = cv2.FastFeatureDetector(30, True)
    kp = fast.detect(input,None)
    output = cv2.drawKeypoints(input, kp, color=(0,0,255))
    return output

# BRIEF key point function
def briefKeyPoint(input):
    star = cv2.FeatureDetector_create("STAR")
    brief = cv2.DescriptorExtractor_create("BRIEF")
    kp = star.detect(input,None)
    kp, des = brief.compute(input, kp)
    output = cv2.drawKeypoints(input,kp,None,(0,0,255),4)
    return output

# ORB key point function
def orbKeyPoint(input):
    orb = cv2.ORB()
    kp = orb.detect(input,None)
    kp, des = orb.compute(input, kp)
    output = cv2.drawKeypoints(input,kp,None,(0,0,255),4)
    return output

# Harris key point function
def HarrisKeyPoint(input):
    input_32 = np.float32(input)
    dst = cv2.cornerHarris(input_32,2,3,0.04)
    output = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
    output[dst>0.01*dst.max()]=[0,0,255]
    return output

# key point match function
def keyPointMatch(input, input_pre):
    input_gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    input_pre_gray = cv2.cvtColor(input_pre, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()
    kp1, des1 = sift.detectAndCompute(input_gray,None)
    kp2, des2 = sift.detectAndCompute(input_pre_gray,None)
    matcher = cv2.DescriptorMatcher_create("FlannBased")
    matches = matcher.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    rows1 = input_gray.shape[0]
    cols1 = input_gray.shape[1]
    rows2 = input_pre_gray.shape[0]
    cols2 = input_pre_gray.shape[1]
    output = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    output[:rows1,:cols1,:] = np.dstack([input_gray, input_gray, input_gray])
    output[:rows2,cols1:cols1+cols2,:] = np.dstack([input_pre_gray, input_pre_gray, input_gray])
    for mat in matches[:5]:
        input_idx = mat.queryIdx
        input_pre_idx = mat.trainIdx
        (x1,y1) = kp1[input_idx].pt
        (x2,y2) = kp2[input_pre_idx].pt
        cv2.circle(output, (int(x1),int(y1)), 4, (0, 0, 255), 1)
        cv2.circle(output, (int(x2)+cols1,int(y2)), 4, (0, 0, 255), 1)
        cv2.line(output, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 0, 255), 3)
    return output, input

# camShift track function
def camShiftTracking(input, roi_hist, term_crit, track_window):
    input_hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([input_hsv],[0],roi_hist,[0,180],1)
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)
    (x,y,w,h) = track_window
    cv2.rectangle(input, (x,y), (x+w,y+h),(0,0,200),2)
    output = input
    return output, track_window

# meanShift track function
def meanShiftTracking(input, roi_hist, term_crit, track_window):
    input_hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([input_hsv],[0],roi_hist,[0,180],1)
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    x,y,w,h = track_window
    cv2.rectangle(input, (x,y), (x+w,y+h),(0,0,200),2)
    output = input
    return output, track_window

# template match function
def templateMaching(input, template):
    input_gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(input_gray,template,eval("cv2.TM_CCOEFF"))
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(input,top_left, bottom_right, (0,0,255), 2)
    output = input
    return output

# Sobel edge extract function
def SobelEdgeExtract(input, dx, dy):
    output = cv2.Sobel(input, 5, dx, dy)
    return output

# Laplacian edge extract function
def LaplacianEdgeExtract(input):
     output = cv2.Laplacian(input,cv2.CV_64F)
     return output

# mosaic process function
def mosaicConvert(input):
    x, y, w, h = 220, 140, 200, 200
    input_mosaic = input[y:y+h, x:x+w]
    input_mosaic = cv2.resize(input_mosaic, (w/5, h/5))
    input_mosaic = cv2.resize(input_mosaic, (w, h), interpolation=cv2.cv.CV_INTER_NN)
    input[y:y+h, x:x+w] = input_mosaic
    output = input
    return output

# scaling process function
def scaleConvert(input, scale):
    h = input.shape[0]
    w = input.shape[1]
    output = cv2.resize(input,(int(w*scale), int(h*scale)))
    return output

# gamma correction function
def gammaCorrection(input, gammaValue):
    input = cv2.pow(input/255.0, gammaValue)
    output = np.uint8(input*255)
    return output

# opening process function
def openingConvert(input):
    k = np.ones((5,5),np.uint8)
    output = cv2.morphologyEx(input, cv2.MORPH_OPEN, k)
    return output

# closing process function
def closingConvert():
    k = np.ones((5,5),np.uint8)
    output = cv2.morphologyEx(input, cv2.MORPH_CLOSE, k)
    return output

# labeling function
def labeling(input):
    input_gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(input_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts = cv2.findContours(th,1,2)[0]
    cv2.drawContours(input,cnts,-2,(255,0,0),-1)
    output = input
    return output

# addine rotation function
def affineConvert(input, rotationDegree):
    rows,cols,ch = input.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),rotationDegree,1)
    output = cv2.warpAffine(input,M,(cols,rows))
    return output

# object extract function
def outlineExtract(input):
    input_th = cv2.adaptiveThreshold(input, 50, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    output = cv2.Canny(input_th, 50, 150, apertureSize = 3)
    return output

# histgram smooth function
def histgramSmooth(input):
    output = cv2.equalizeHist(input)
    return output

# optical flow calculate function
def opticalFlowCalc(input, input_pre, input_hsv):
    flow = cv2.calcOpticalFlowFarneback(input, input_pre, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    input_hsv[...,0] = ang*180/np.pi/2
    input_hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    output = cv2.cvtColor(input_hsv,cv2.COLOR_HSV2BGR)
    return output, input

# face detect function
def faceDetect(input, faceCascade):
    face = faceCascade.detectMultiScale(input, 1.1, 3)
    for (x, y, w, h) in face:
        cv2.rectangle(input, (x, y),(x + w, y + h),(0, 50, 255), 3)
    output = input
    return output

# human detect function
def fullBodyDetect(input):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    hogParams = {'winStride': (8, 8), 'padding': (32, 32), 'scale': 1.05}
    human, r = hog.detectMultiScale(input, **hogParams)
    for (x, y, w, h) in human:
        cv2.rectangle(input, (x, y),(x+w, y+h),(0,50,255), 3)
    output = input
    return output

# motion detect function
def motionDetect(input, input_pre):
    input_gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    input_pre_gray = cv2.cvtColor(input_pre, cv2.COLOR_BGR2GRAY)
    (w,h) = (input.shape[0],input.shape[1])
    output = np.zeros((w,h),np.uint8)
    diff = cv2.absdiff(input_gray,input_pre_gray)
    fg = diff > 10
    output[fg] = 255
    return output, input

# binarization function
def binaryConvert(input):
    ret, output = cv2.threshold(input, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return output

# color tracking function
def colorTracking(input):
    input_hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
    color_min = np.array([25,50,50])
    color_max = np.array([35,255,255])
    color_mask = cv2.inRange(input_hsv, color_min, color_max)
    output = cv2.bitwise_and(input, input, mask=color_mask)
    return output

# Hough convert function
def Houghconvert(input):
    input_th = cv2.adaptiveThreshold(input, 50, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    input_edge = cv2.Canny(input_th, 50, 150, apertureSize = 3)
    lines = cv2.HoughLines(input_edge, 2, np.pi/180,200)
    circles = cv2.HoughCircles(input,cv2.HOUGH_GRADIENT,1,17,param1=110,param2=120,minRadius=51,maxRadius=900)
    circles = np.uint16(np.around(circles))
    for rho,theta in lines[0]:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a*rho,b*rho
        x1, y1 = int(x0 + 1000*(-b)), int(y0 + 1000*(a))
        x2, y2 = int(x0 - 1000*(-b)), int(y0 - 1000*(a))
        cv2.line(input,(x1, y1), (x2, y2), (0, 0, 255), 2)
    for i in circles[0,:]:
        cv2.circle(input,(i[0],i[1]),i[2],(255,0,0),5)
        cv2.circle(input,(i[0],i[1]),2,(0,255,0),5)
    output = input
    return output

# foreground extract function
def foregroundExtract(input):
    x1,y1,x2,y2 = 0,0,500,350
    mask = np.zeros(input.shape[:2],np.uint8)
    bgd_model = np.zeros((1,65),np.float64)
    fgd_model = np.zeros((1,65),np.float64)
    rect = (x1,y1,x2,y2)
    cv2.grabCut(input,mask,rect,bgd_model,fgd_model,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype("uint8")
    output = input*mask2[:,:,np.newaxis]
    return output

# sharpening filter function
def sharpeningFilter(input):
    kernel = np.array([[0,-1,0],
                       [-1,5,-1],
                       [0,-1,0] ],np.float32)
    output = cv2.filter2D(input,-1,kernel)
    return output

# embossment filter function
def embossmentFilter(input):
    kernel = np.array([[-1,-1,0],
                       [-1,1,1],
                       [0,1,0] ],np.float32)
    output = cv2.filter2D(input,-1,kernel)
    return output

# decrease color function
def decreaseColor(input):
    Z = input.reshape((-1,3))
    Z = np.float32(Z)
    K = 8
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    output = res.reshape((input.shape))
    return output

# write HSV function
def writeHSV(input, x, y):
    input_hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
    if(cx!=0):
        (x,y) = (cx, cy)
    h,s,v = input_hsv[y,x]
    cv2.putText(input,"H:"+str(h),(20,40),2,1,(0,0,200))
    cv2.putText(input,"S:"+str(s),(180,40),2,1,(0,200,0))
    cv2.putText(input,"V:"+str(v),(340,40),2,1,(200,0,0))
    cv2.rectangle(input,(x-5,y-5),(x+5,y+5),(0, 50, 255), 2)
    output = input
    return output, x, y

# write RGB function
def writeRGB(input, x, y):
    if(cx!=0):
        (x,y) = (cx, cy)
    b,g,r = input[y,x]
    cv2.putText(input,"Red:"+str(r),(20,80),2,1,(0,0,200))
    cv2.putText(input,"Green:"+str(g),(180,80),2,1,(0,200,0))
    cv2.putText(input,"Blue:"+str(b),(340,80),2,1,(200,0,0))
    cv2.rectangle(input,(x-5,y-5),(x+5,y+5),(0, 50, 255), 2)
    output = input
    return output, x, y

# moment calculate function
def momentCalc(input):
    input_gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(input_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts = cv2.findContours(th,1,2)[0]
    areas = [cv2.contourArea(cnt) for cnt in cnts]
    cnt_max = [cnts[areas.index(max(areas))]][0]
    M = cv2.moments(cnt_max)
    (cx, cy) = ( int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]) )
    cv2.circle(input,(cx,cy),5, (0,0,255), -1)
    output = input
    return output

# phase correlate function
def phaseCorrelation(input, input_pre):
    input_32 = np.float32(input)
    input_pre_32 = np.float32(input_pre)
    dx, dy = cv2.phaseCorrelate(input_32,input_pre_32)
    return dx, dy

# power spectrum function
def powerSpecrum(input):
    fshift = np.fft.fftshift(np.fft.fft2(input))
    fft = 20*np.log(np.abs(fshift))
    fft = cv2.normalize(fft, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return fft

# main function
def main():
    # =====Initialize windows=====
    cv2.namedWindow('Window1', cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow('Window1', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Window1', 1920, 1280)

    # =====Iitialize cameras=====
    cam = cv2.VideoCapture(0)
    if (cam.isOpened() == False):
        print 'Could not open camera'

    # =====Initialize CamShift and meanShift=====
    ret, frame = cam.read()
    (r,h,c,w) = (200,80,280,80)
    track_window = (c,r,w,h)
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0, 60,32)), np.array((180,255,255)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    # =====Initialize key point match and motion detect=====
    frame_pre = frame

    # =====Initialize optical flow=====
    frame_gray_pre = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame)
    hsv[...,1] = 255

    # =====machine learning=====
    faceFileLoc = r'C:\Users\Precision M3800\Documents\opencv\sources\data\haarcascades\haarcascade_frontalface_alt.xml'
    faceCascade = cv2.CascadeClassifier(faceFileLoc)

    # =====Initialize mouse action=====
    (x, y) = (6,6)

    # =====template matching=====
    templateLoc = r'C:\Users\Precision M3800\Documents\Portable Python\Image\PSvita.jpg'
    template = cv2.imread(templateLoc)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # =====Loop=====
    while (True):
        ret, frame = cam.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.setMouseCallback("Window1",onMouse)
#===============================================================================
        #frame = surfKeyPoint(frame_gray)
        #frame = siftKeyPoint(frame_gray)
        #frame = ShiTomasiKeyPoint(frame_gray)
        #frame = fastKeyPoint(frame_gray)
        #frame = briefKeyPoint(frame_gray)
        #frame = orbKeyPoint(frame_gray)
        #frame = HarrisKeyPoint(frame_gray)
        #frame, frame_pre = keyPointMatch(frame, frame_pre)
        #frame, track_window = camShiftTracking(frame, roi_hist, term_crit, track_window)
        #frame, track_window = meanShiftTracking(frame, roi_hist, term_crit, track_window)
        #frame = templateMaching(frame, template_gray)
        #frame = SobelEdgeExtract(frame, 0, 1)
        #frame = LaplacianEdgeExtract(frame)
        #frame = mosaicConvert(frame)
        #frame = scaleConvert(frame, 1.5)
        #frame = gammaCorrection(frame, 0.8)
        #frame = openingConvert(frame)
        #frame = closingConvert(frame)
        #frame = labeling(frame)
        #frame = affineConvert(frame, 45)
        #frame = outlineExtract(frame_gray)
        #frame = histgramSmooth(frame_gray)
        #frame, frame_gray_pre = opticalFlowCalc(frame_gray, frame_gray_pre, hsv)
        #frame = faceDetect(frame, faceCascade)
        #frame = fullBodyDetect(frame)
        #frame, frame_pre = motionDetect(frame, frame_pre)
        #frame = binaryConvert(frame_gray)
        #frame = colorTracking(frame)
        #frame = Houghconvert(frame_gray)
        #frame = foregroundExtract(frame)
        #frame = sharpeningFilter(frame)
        #frame = embossmentFilter(frame)
        #frame = decreaseColor(frame)
        #frame, x, y = writeHSV(frame, x, y)
        #frame, x, y = writeRGB(frame, x, y)
        #frame = momentCalc(frame)
        #dx,dy = phaseCorrelation(frame_gray, frame_gray_pre)
        #frame = powerSpecrum(frame_gray)
#===============================================================================
        cv2.imshow('Window1',frame)
        key = cv2.waitKey(33)
        if (key == 27):
            break

    # =====destructor=====
    cam.release()
    cv2.destroyAllWindows()

# procedure
if __name__ == '__main__':
    main()
