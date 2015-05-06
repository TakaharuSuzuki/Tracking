#-------------------------------------------------------------------------------
# Name:        colorTracking
# Purpose:
#
# Author:      Takaharu Suzuki
#
# Created:     31/03/2015
# Copyright:   (c) Precision M3800 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import cv2

def nothing(x):
    pass

def main():
    cv2.namedWindow('Window0', cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow('Window1', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Window2', cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow('Window3', cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow('Window4', cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow('Window5', cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow('Window6', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Window7', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Window8', cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow('Window9', cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow('Window10', cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow('Window11', cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow('Window12', cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow('Window13', cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow('Window14', cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow('Window15', cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar("H_min","Window0",1,180,nothing)
    cv2.createTrackbar("S_min","Window0",1,255,nothing)
    cv2.createTrackbar("V_min","Window0",1,255,nothing)
    cv2.createTrackbar("H_max","Window0",180,180,nothing)
    cv2.createTrackbar("S_max","Window0",255,255,nothing)
    cv2.createTrackbar("V_max","Window0",255,255,nothing)
    cv2.createTrackbar("Blur","Window0",1,50,nothing)
    cv2.createTrackbar("Iteration","Window0",1,100,nothing)

    cam = cv2.VideoCapture(0)
    if (cam.isOpened() == False):
        print 'Could not open camera'

    while (True):
        ret, frame = cam.read()
        h_min = cv2.getTrackbarPos("H_min","Window0")
        s_min = cv2.getTrackbarPos("S_min","Window0")
        v_min = cv2.getTrackbarPos("V_min","Window0")
        h_max = cv2.getTrackbarPos("H_max","Window0")
        s_max = cv2.getTrackbarPos("S_max","Window0")
        v_max = cv2.getTrackbarPos("V_max","Window0")
        blur = cv2.getTrackbarPos("Blur","Window0")
        iteration = cv2.getTrackbarPos("Iteration","Window0")
        #cv2.imshow('Window1',frame)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_min = np.array([h_min,s_min,v_min])
        color_max = np.array([h_max,s_max,v_max])
        color_mask = cv2.inRange(frame_hsv, color_min, color_max)
        frame_color = cv2.bitwise_and(frame, frame, mask=color_mask)
        cv2.imshow('Window2',frame_color)
        frame_gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('Window3',frame_gray)
        frame_smooth = cv2.medianBlur(frame_gray,blur*2+1)
        #cv2.imshow('Window4',frame_smooth)
        ret, frame_binary = cv2.threshold(frame_smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #cv2.imshow('Window5',frame_binary)
        frame_morph = cv2.morphologyEx(frame_binary, cv2.MORPH_OPEN, None, iterations=iteration)
        #cv2.imshow('Window6',frame_morph)
        cnts = cv2.findContours(frame_morph,1,2)[0]
        areas = [cv2.contourArea(cnt) for cnt in cnts]
        if areas:
            cnt_max = [cnts[areas.index(max(areas))]][0]
            cv2.drawContours(frame_morph,[cv2.convexHull(cnt_max)],0,255,-1)
        cv2.imshow('Window7',frame_morph)
        if areas:
            cv2.drawContours(frame,[cv2.convexHull(cnt_max)],0,(255,0,0),10)
            M = cv2.moments(cnt_max)
            (cx, cy) = ( int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]) )
            cv2.circle(frame,(cx,cy),10, (0,0,255), -1)
        cv2.imshow('Window8',frame)

        #frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #color_min = np.array([33,100,100])
        #color_max = np.array([35,255,255])
        #color_mask = cv2.inRange(frame_hsv, color_min, color_max)
        #frame_color = cv2.bitwise_and(frame, frame, mask=color_mask)
        #cv2.imshow('Window9',frame_color)
        #frame_gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('Window10',frame_gray)
        #frame_smooth = cv2.medianBlur(frame_gray,51)
        #cv2.imshow('Window11',frame_smooth)
        #ret, frame_binary = cv2.threshold(frame_smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #cv2.imshow('Window12',frame_binary)
        #frame_morph = cv2.morphologyEx(frame_binary, cv2.MORPH_OPEN, None, iterations=10)
        #cv2.imshow('Window13',frame_morph)
        #cnts = cv2.findContours(frame_morph,1,2)[0]
        #areas = [cv2.contourArea(cnt) for cnt in cnts]
        #if areas:
        #    cnt_max = [cnts[areas.index(max(areas))]][0]
        #    cv2.drawContours(frame_morph,[cv2.convexHull(cnt_max)],0,255,-1)
        #cv2.imshow('Window14',frame_morph)
        #if areas:
        #    cv2.drawContours(frame,[cv2.convexHull(cnt_max)],0,(255,0,0),10)
        #    M = cv2.moments(cnt_max)
        #    (cx, cy) = ( int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]) )
        #    cv2.circle(frame,(cx,cy),10, (0,0,255), -1)
        #cv2.imshow('Window15',frame)


        key = cv2.waitKey(33)
        if (key == 27):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
