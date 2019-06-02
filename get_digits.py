#!/usr/bin/env python2.7
# coding=utf-8
#
# Read digital seven-segment lcd screen and update this as heat meter. The 
# meter has a resolution of 1 megajoule. At high heating (±5kW), this is 
# is consumed in 3 minutes, so we read and update every 2 minutes (720/day)
#
# How to run / debug
# 
# Run as ipython --pdb -- ./get_digits.py --roi 457 727 1268 706 1273 521 463 537
# Does not with in python3.6 -> Ipython.embed. Run directly from ipython-3.6
# Fix using  globals().update( locals() );

# Based on https://www.pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/

import io
import os
import sys
import cv2
import re
import requests
import numpy as np

from datetime import datetime
import time
import argparse
import logging

# from IPython import embed;


# Python2/3 compatibility
try:
   input = raw_input
except NameError:
   pass

def rotateImage(image, angle):
    """
    From https://stackoverflow.com/a/9042907
    Required because not all opencv versions have cv2.rotate
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def kMeans(X, K, maxIters = 10):
    # from https://gist.github.com/bistaumanga/6023692
    # Take two random samples from array
    centroids = X[np.random.choice(np.arange(len(X)), K)]

    for i in range(maxIters):
        # Cluster Assignment step
        C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
        
        # Ensure we have K clusters, otherwise reset centroids. If we don't, 
        # results might be nan
        if (len(np.unique(C)) < K):
            centroids = X[np.random.choice(np.arange(len(X)), K)]
        else:
            # Move centroids step 
            centroids = [X[C == k].mean(axis = 0) for k in range(K)]
    return np.array(centroids) , C

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
 
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
 
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
 
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
 
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
 
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
 
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
    # return the warped image
    return warped

def calibrate_image(im_path, ndigit, rotate=None, roi=None, digwidth=None, segwidth=None, segthresh=None, debug=False):
    image = cv2.imread(im_path)
    if (rotate):
        image = rotateImage(image, rotate)
    opt_str = ""

    import pylab as plt
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    while np.any(roi==None):
        # Show image, mirror vertically because image and graph orientation 
        # differ
        plt.figure(10)
        plt.clf()
        plt.imshow(image, origin='upper')

        # get ROI from user, convert to array
        im_roi_str = input("ROI? (x0 y0 x1 y1 x2 y2 x3 y3)")
        im_roi_tmp = np.r_[im_roi_str.split(" ")].astype(int).reshape(-1,2)

        warped = four_point_transform(image, im_roi_tmp)

        plt.figure(20)
        plt.clf()
        plt.imshow(warped, origin='upper')
        if (input("ROI OK? y/n") == "y"):
            roi = im_roi_tmp
            opt_str += "--roi {} ".format(im_roi_str)
    
    img_norm, img_thresh = preproc_img(im_path, None, roi, rotate=rotate, debug=debug)

    while not digwidth:
        plt.figure(100)
        plt.clf()
        plt.imshow(img_thresh, origin='upper')

        # get input from user
        dig_width_str = input("digit width in pix? (width)")
        dig_width_tmp = int(dig_width_str)
        dig_margin = (img_thresh.shape[1] - dig_width_tmp) / (ndigit - 1) - dig_width_tmp

        # Highlight digits
        ax1 = plt.gca()
        for i in range(ndigit):
            ax1.add_patch(
            patches.Rectangle(
                (i*(dig_width_tmp+dig_margin), 0),   # (x,y)
                dig_width_tmp,          # width
                img_thresh.shape[0],            # height
                fill=True, color='k', alpha=0.3)
            )

        if (input("Digit size OK? y/n") == "y"):
            digwidth = dig_width_tmp
            opt_str += "--digwidth {} ".format(dig_width_str)

    dig_margin = (img_thresh.shape[1] - digwidth) / (ndigit - 1) - digwidth

    while not segwidth:
        plt.figure(200)
        plt.clf()
        plt.imshow(img_thresh, origin='upper')

        # get input from user
        seg_width_str = input("segment width in pix? (width)")
        print(seg_width_str)
        seg_width_tmp = int(seg_width_str)

        # Highlight digits and segments
        ax1 = plt.gca()

        dM = dH = dW = seg_width_tmp
        dHC = dW//2
        w, h = digwidth, img_thresh.shape[0]

        segments = calc_seg_coordinates(w,h,seg_width_tmp)
        #     [
        #     ((dM, 0), (w-2*dM, dH))                 # top
        #     ,((0, dM), (dW, (h // 2)-int(1.5*dM)))         # top-left
        #     ,((w - dW, dM), (dW, (h // 2)-int(1.5*dM)))    # top-right
        #     ,((dM, (h // 2) - dHC) , (w-2*dM, dH))  # center
        #     ,((0, (h // 2)+int(0.5*dM)), (dW, (h//2)-int(1.5*dM)))  # bottom-left
        #     ,((w - dW, (h // 2)+int(0.5*dM)), (dW, (h//2)-int(1.5*dM)))    # bottom-right
        #     ,((dM, h - dH), (w-2*dM, dH))           # bottom
        # ]

        for i in range(ndigit):
            xdig = i*(digwidth+dig_margin)
            ydig = 0
            for s_orig, s_size in segments:
                ax1.add_patch(
                patches.Rectangle(
                    (xdig + s_orig[0], ydig + s_orig[1]),   # (x,y)
                    s_size[0],          # width
                    s_size[1],            # height
                    fill=False, color='k')
                )

        if (input("Segment width OK? y/n") == "y"):
            segwidth = seg_width_tmp
            opt_str += "--segwidth {} ".format(seg_width_str)

    while not segthresh:
        digit_levels = read_digits(img_thresh, ndigit, digwidth, segwidth)
        plt.figure(300)
        plt.clf()
        plt.plot(np.r_[digit_levels], '.')

        centroids, C = kMeans(np.r_[digit_levels].flatten(), K=2)
        plt.hlines(centroids.mean(), 0, len(digit_levels))

        seg_thresh_str = input("Segment fill threshold in fraction? (k-means avg: {})".format(centroids.mean()))
        print(seg_thresh_str)
        seg_thresh_tmp = float(seg_thresh_str)

        plt.hlines(seg_thresh_tmp, 0, len(digit_levels))

        if (input("Segment fill threshold OK? y/n") == "y"):
            segthresh = seg_thresh_tmp
            opt_str += "--segthresh {} ".format(seg_thresh_str)

    if (input("Try to maximize k-means avg through digit and segment width? y/n") == "y"):
        digwidth_arr = np.arange(digwidth*0.9, digwidth*1.1, dtype=int)
        segwidth_arr = np.arange(segwidth*0.8, segwidth*1.2, dtype=int)
        # print(digwidth_arr)
        # print(segwidth_arr)
        # For a series of digit width and segment widths, calculate segment 
        # filling delta between ON and OFF segments. This should be as high 
        # as possible.
        # NB this assumes the kMeans division above succeeded!
        # NB this only gives results for current image
        kmean_mat = [[np.r_[read_digits(img_thresh, ndigit, dw, sw)].flatten()[np.argwhere(C==0)].mean()-np.r_[read_digits(img_thresh, ndigit, dw, sw)].flatten()[np.argwhere(C==1)].mean() for dw in digwidth_arr] for sw in segwidth_arr]
        #kmean_mat = [[kMeans(np.r_[read_digits(img_thresh, ndigit, dw, sw)].flatten(), K=2)[0].ptp() for dw in digwidth_arr] for sw in segwidth_arr]
        # print(kmean_mat)
        plt.imshow(kmean_mat, extent=[digwidth_arr.min()-0.5, digwidth_arr.max()+0.5, segwidth_arr.min()-0.5, segwidth_arr.max()+0.5])

    lcd_digit_levels = read_digits(img_thresh, ndigit, digwidth, segwidth, debug)
    lcd_value, lcd_probability = calc_value(lcd_digit_levels, segthresh)

    input("Got reading: {}, press any key to quit".format(lcd_value, lcd_probability))

    return opt_str

def capture_img(delay=2, method='data'):
    # Only import here, we don't have this lib on dev computer
    from picamera import PiCamera
    logging.debug("Acquiring image, method={}...".format(method))

    if (method == 'data'):
        # Create the in-memory stream
        stream = io.BytesIO()
        with PiCamera() as camera:
            # TODO: how to run at native resolution?
            camera.resolution = (2592, 1944)
            camera.framerate = 15
            logging.debug("Camera warming up {}s...".format(delay))
            time.sleep(delay)
            camera.capture(stream, format='jpeg')

        # Construct a numpy array from the stream
        data = np.fromstring(stream.getvalue(), dtype=np.uint8)
        # "Decode" the image from the array, preserving colour
        image = cv2.imdecode(data, 1)
        return None, image

    else:
        # Via file
        camera = PiCamera()
        # TODO: how to run at native resolution?
        camera.resolution = (2592, 1944)
        camera.framerate = 15

        logging.debug("Camera warming up {}s...".format(delay))
        time.sleep(delay)
        #camera.awb_mode='off'
        #camera.exposure_mode = 'off'
        img_path = 'capture_{}.jpg'.format(time.time())
        camera.capture(img_path, use_video_port=False)
        return img_path, None

def preproc_img(imgpath, image, roi, rotate=None, store_crop=False, debug=False):
    logging.debug("Pre-processing image")
    # Read image from disk, select ROI, convert to grayscale
    if (not imgpath == None):
        image = cv2.imread(imgpath)

    if (rotate):
        image = rotateImage(image, rotate)

    warped = four_point_transform(image, roi)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    if (debug):
        plt.figure(100)
        plt.clf()
        plt.title('gray')
        plt.imshow(gray, origin='upper')

    # Normalize by dividing by a blurred version, use 90% of the height of 
    # the image as kernel for this
    blurred = cv2.GaussianBlur(gray, (int(warped.shape[0]*0.9), int(warped.shape[0]*0.9)), 0)
    norm = gray / blurred
    norm = cv2.normalize(norm, norm, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if (store_crop):
        # TODO: store to log directory, not just anywhere (defaults to 
        # homedir now)
        try:
            fname = 'capture_crop_{}.jpg'.format(time.time())
            fpath = os.path.join(store_crop, fname)
            cv2.imwrite(fpath, norm)
        except Exception as inst:
            logging.warn("Could not store crop: {}".format(inst))


    if (debug):
        plt.figure(110)
        plt.clf()
        plt.title('normalized')
        plt.imshow(norm, origin='upper')

    # Binarize image
    # https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html
    thresh = cv2.threshold(norm, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    if (debug):
        plt.figure(111)
        plt.clf()
        plt.title('threshold')
        plt.imshow(thresh, origin='upper')


    # adaptiveThreshold doesnt work well for this situation
    # thresh2 = cv2.adaptiveThreshold(norm, 255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, int(warped.shape[0]*0.1), 1)

    # See https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh2 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Optional: dilate once more to fill segments better
    #thresh2 = cv2.dilate(thresh, cv2.MORPH_OPEN, np.ones((3,3)))

    if (debug):
        plt.figure(120)
        plt.clf()
        plt.title('morphed')
        plt.imshow(thresh2, origin='upper')

    return norm, thresh2

def read_digits(img, ndigit, digwidth, segwidth, debug=False):
    logging.debug("Reading digits...")
    # img: pre-processed image
    # ndigit: number of digits (numbers) to extract
    # digwidth: width of one digit (without margin) in pixels
    # segwidth: width of one of seven segments per digiti (without space), in pixels
    
    ## Extract digits
    # margin between digits is given by:
    dig_margin = (img.shape[1] - digwidth) / (ndigit - 1) - digwidth
    w, h = digwidth, img.shape[0]

    if (debug):
        plt.figure(30)
        plt.clf()
        plt.imshow(img, origin='upper')

    digit_levels = []
    for i in range(ndigit):
        # ROI of each digit, counter clockwise start bottom-left
        dig_roi = np.r_[ [
            [ i*(w+dig_margin)  , 0],
            [ i*(w+dig_margin)+w, 0],
            [ i*(w+dig_margin)+w, h],
            [ i*(w+dig_margin)  , h] ]]
        # Ensure our ROI is not outside the image (only needed for 4pt trans)
        digit = img[0:h, int(i*(w+dig_margin)) : int(i*(w+dig_margin)+w) ]
        if (debug):
            plt.figure(40+i)
            plt.clf()
            plt.imshow(digit, origin='upper')
        digit_levels.append( read_segments(digit, segwidth, debug) )
    
    return digit_levels

def calc_seg_coordinates(digitw, digith, segwidth):
    w, h = digitw, digith
    dM = dH = dW = segwidth
    dHC = dW//2

    return [
        ((dM, 0), (w-2*dM, dH)),                 # top
        ((0, dM), (dW, (h // 2)-int(1.5*dM))),   # top-left
        ((w - dW, dM), (dW, (h // 2)-int(1.5*dM))),    # top-right
        ((dM, (h // 2) - dHC) , (w-2*dM, dH)),  # center
        ((0, (h // 2)+int(0.5*dM)), (dW, (h//2)-int(1.5*dM))),  # bottom-left
        ((w - dW, (h // 2)+int(0.5*dM)), (dW, (h//2)-int(1.5*dM))),    # bottom-right
        ((dM, h - dH), (w-2*dM, dH))           # bottom
    ]

def read_segments(digit_img, segwidth, debug=False):
    # Given an image of a digit, and segwidth in pixels, return segment 
    # states

    # Segment coordinates (in pixels as (x,y), (w, h))
    dM = dH = dW = segwidth
    dHC = dW//2
    w, h = digit_img.shape[1], digit_img.shape[0]
    segments = calc_seg_coordinates(w,h,segwidth)
    # [
    #     ((dM, 0), (w-2*dM, dH))                 # top
    #     ,((0, dM), (dW, (h // 2)-1*dM))         # top-left
    #     ,((w - dW, dM), (dW, (h // 2)-1*dM))    # top-right
    #     ,((dM, (h // 2) - dHC) , (w-2*dM, dH))  # center
    #     ,((0, (h // 2)+dM), (dW, (h//2)-1*dM))  # bottom-left
    #     ,((w - dW, (h // 2)+dM), (dW, (h//2)-1*dM))    # bottom-right
    #     ,((dM, h - dH), (w-2*dM, dH))           # bottom
    # ]
    
    seg_level = []
    for (i, ((xA, yA), (xlen, ylen))) in enumerate(segments):
        # extract the segment ROI, count the total number of
        # thresholded pixels in the segment, and then compute
        # the area of the segment
        seg = digit_img[yA:yA+ylen, xA:xA+xlen]
        #total = seg.sum()/255.
        #area = float(seg.shape[0] * seg.shape[1])
        seg_level.append( (seg.mean()/255) )
    return seg_level
    
def calc_value(digit_levels, segthresh, minval=0, maxinc=None):
    """
    Given intensity levels of all segments, calculate most likely value
    per digit for the whole number.
    """

    # This is what the seven segments look like for each number
    DIGITS_MATCH = np.r_[
        # 0  1  2  3  4  5  6
        [[1, 1, 1, 0, 1, 1, 1]], # 0
        [[0, 0, 1, 0, 0, 1, 0]], # 1
        [[1, 0, 1, 1, 1, 0, 1]], # 2
        [[1, 0, 1, 1, 0, 1, 1]], # 3
        [[0, 1, 1, 1, 0, 1, 0]], # 4
        [[1, 1, 0, 1, 0, 1, 1]], # 5
        [[1, 1, 0, 1, 1, 1, 1]], # 6
        [[1, 0, 1, 0, 0, 1, 0]], # 7 This is the 3-segment seven (vs 4-seg)
        [[1, 1, 1, 1, 1, 1, 1]], # 8
        [[1, 1, 1, 1, 0, 1, 1]] # 9
    ]
    ndigit = len(digit_levels)

    # Per digit, we have the intensity level for each segment. We calculate the 
    # difference of these segment intensities with all the template value (DIGITS_MATCH).
    # The lower the difference, the better the match and the more likely this digit
    # has a specific value.
    digit_dist = [] 
    digit_candidates = []
    for digit_level_v in digit_levels:
        # Squared difference distance as matrix, then sum over segments
        c = (((DIGITS_MATCH*2*segthresh) - digit_level_v)**2.0).sum(1)
        # The index of a match is also the digit. If we sort the argsort the distance 
        # array, we get a list of digit candidates
        c_ord = np.argsort(c)
        
        digit_dist.append(list(c[c_ord]))
        digit_candidates.append(list(c_ord))

    # Flatten list of digit candidates, ordered by probability, and paired with digit index
    digit_dist = np.r_[digit_dist]
    cand_id_next = np.unravel_index(np.argsort(digit_dist, axis=None), digit_dist.shape)

    # The most likely match per digit might not be the best match overall. We 
    # check a range of possible numbers below that might satisfy higher-level
    # numerical constraints.

    # Example for 3 digits: 
    # Try candidate [0,0,0] first, which has digits 
    # d0=digit_candidates[0][0], d1=digit_candidates[1][0], d2=digit_candidates[2][0]
    # The probability of this number is:
    # digit_dist[0][0] + digit_dist[1][0] + digit_dist[2][0]
    # The value of which is
    # v = d0*10^2 + d1*10^1 + d0*10^0
    #
    # If v is not ok numerically, try the second candidate, which might be
    # [0, 0, 1], i.e. use the second match for the third digit (from cand_id_next)

    # We try the most likely digit candidate first (which has index 0)
    cand_id = ndigit*[0]
    
    # Try 2 candidates for each digit (i.e. 14 for 7 digit display)
    for off in range(ndigit*2):
        # Build numerical value from digit candidates for the first candidate
        # N.B. Important: convert to int() first which can support arbitrarily large 
        # values, else the numpy default (int32) is used and we overflow above 
        # log10(2**31) = 10 digits
        cand = sum([int(digit_candidates[i][c])*10**(ndigit-i-1) for i, c in enumerate(cand_id)])
        # Probability of this total match
        prob = sum([digit_dist[i][c] for i, c in enumerate(cand_id)])
        logging.debug("Found candidate value {} with distance {:.4}...".format(cand, prob))
        
        # Check if candidate number satisfies numerical constraints
        if (cand >= minval and (not maxinc or cand < (minval + maxinc))):
            return cand, prob

        # Candidate was not ok, update next most probable digit and retry
        cand_id[cand_id_next[0][off+ndigit]] = cand_id_next[1][off+ndigit]
    
    # If we exit the loop above we found no match that satisfies numerical 
    # constraints (not good!). Try to salvage by returning most likely 
    # candidate
    cand_id = ndigit*[0]
    cand0 = sum([digit_candidates[i][c]*10**(ndigit-i-1) for i, c in enumerate(cand_id)])
    prob0 = sum([digit_dist[i][c] for i, c in enumerate(cand_id)])
    logging.warning("Could not satisfy range constraints, returning most likly result {}...".format(cand0))
    return cand0, prob0

def domoticz_init(ip, port, meter_idx, prot="http"):
    # Get current water meter reading from domoticz, return meter_count_l

    logging.debug("Get meter {} reading from domoticz".format(meter_idx))
    
    # E.g. https://127.0.0.1:10443/json.htm?type=devices&rid=
    req_url = "{}://{}:{}/json.htm?type=devices&rid={}".format(prot, ip, port, meter_idx)

    try:
        resp = requests.get(req_url, verify=False, timeout=5)
    except requests.exceptions.Timeout as inst:
        logging.error("Could not get current meter reading due to timeout: {}, failing".format(inst))
        raise
    except Exception as inst:
        logging.error("Could not get current meter reading: {}, failing".format(inst))
        raise

    # Get meter offset ('AddjValue'), given as float
    offset_str = resp.json()['result'][0]['AddjValue'] # like '13.456'
    offset_l = int(float(offset_str)*1000)

    # Get counter value, which domoticz gives crappy string with or without 
    # units and with or without space (I mean why have an API at all?), more 
    # specically: 
    # [0-9]+\.?[0-9]*\( )?(kWh|m3), e.g. '13.4 m3' or '0 m3'
    count_str_raw = resp.json()['result'][0]['Data']
    count_str = re.search(r'(\d+\.?\d*)', count_str_raw).group(1)
    count_l = int(float(count_str)*1000)

    # Get last update time
    last_str = resp.json()['result'][0]['LastUpdate'] # like '13.456'
    last_t = datetime.strptime(last_str, '%Y-%m-%d %H:%M:%S')
    last_delay = datetime.now() - last_t

    logging.info("Meter {}: counter={}, offset={}, delay={}".format(meter_idx, count_l, offset_l, last_delay))
    return offset_l, count_l, last_delay

def domoticz_update(value, prot='https', ip='127.0.0.1', port='443', m_idx=None):
    # Value is always in MJ (meter shows GJ with 3 decimal digits, we ignore 
    # the digit, leaving value in MJ. Domoticz meters are in kWh and m^3 gas
    # For heat, we have 3581.92 m^3 == 271.199 GJ (according to our own 
    # meter), or 13.20772 m^3/GJ 
    # for kWh we have 277.7777 kWh/GJ
    val_MJ = value
    val_millim3 = val_MJ * 13.20772
    #val_Wh = val_MJ * 277.7777

    # Get current value and offset
    #m_idx = 28
    m_offs, m_count, m_delay = domoticz_init(ip, port, m_idx, prot)
    # We use an incremental meter, only update the value minus the count
    upd_val_millim3 = int(val_millim3 - m_count)

    # Do not update domoticz below 10 units because of domoticz resolution being too small, causing flip flopping updates
    if (upd_val_millim3 < 10):
        logging.info("Not updating meter {}, value {} too small for domoticz resolution.".format(m_idx, upd_val_millim3))
    # Rate limit to max 2 GJ/day = 25000 millim^3/day to prevent wrong reads = val_MJ > 1000*m_delay.total_seconds()/3600/24
    elif (upd_val_millim3 > 25000*m_delay.total_seconds()/3600/24):
        logging.info("Not updating meter {}, value {} > {}.".format(m_idx, upd_val_millim3, 25000*m_delay.total_seconds()/3600/24))
    else:
        req_url = "{}://{}:{}/json.htm?type=command&param=udevice&idx={}&svalue={}".format(prot, ip,port, m_idx, int(upd_val_millim3))
        logging.info("Updating meter {} to value {}".format(m_idx, upd_val_millim3))
        try:
            httpresponse = requests.get(req_url, verify=False, timeout=5)
        except requests.exceptions.Timeout as inst:
            logging.warn("Could not update meter reading due to timeout: {}, failing".format(inst))

    ## Update power in kWh and W
    ## This does not work (yet) because domoticz JSON api does not print in 
    # constant precision, for small numbers we get 3 decimals, for large 
    # numbers none. This makes power difficult to calculate. Crappy JSON 
    # interface again

    # m_idx = 27
    # # Get current value and offset
    # m_offs, m_count, m_delay = domoticz_init(ip, port, m_idx, prot)
    # # We use an absolute meter, update full value
    # upd_val_Wh = int(val_Wh)
    # # Power in Wh*3600 / (time in s)
    # upd_val_power = int(3600 * (val_Wh - m_count) / m_delay.total_seconds())

    # req_url = "{}://{}:{}/json.htm?type=command&param=udevice&idx={}&svalue={};{}".format(prot, ip,port, m_idx, int(upd_val_power), int(upd_val_Wh))
    # logging.info("Updating meter {} to value {};{}".format(m_idx, upd_val_power, upd_val_Wh))
    # httpresponse = requests.get(req_url, verify=False, timeout=5)

def influxdb_update(value, prot='http', ip='127.0.0.1', port='8086', db="smarthome", query="energy,type=heat,device=landisgyr value="):
    """
    Push update to influxdb with second precision

    Value is in MJ (meter shows GJ with 3 decimal digits, we ignore 
    the digit, leaving value in MJ). We convert to Joule to get SI in influxdb
    """
    value_joule = value*1000000
    
    # Something like req_url = "http://localhost:8086/write?db=smarthometest&precision=s"
    req_url = "{}://{}:{}/write?db={}&precision=s".format(prot, ip, port, db)
    # Something like post_data = "energy,type=heat,device=landisgyr value=10"
    # Alternatively, like post_data = "energy landisgyr=10"
    post_data = "{}{:d}".format(query, int(value_joule))

    logging.info("Pushing data '{}' to influxdb".format(post_data))

    try:
        httpresponse = requests.post(req_url, data=post_data, verify=False, timeout=5)
    except requests.exceptions.Timeout as inst:
        logging.warn("Could not update meter reading due to timeout: {}, failing".format(inst))
        pass

def main():
    parser = argparse.ArgumentParser(description='Read seven-segment LCD display.')
    parser.add_argument('--ndigit', type=int, metavar='N', required=True,
                        help='number of digits in the ROI')

    parser.add_argument('--rotate', metavar='deg', type=int,
                        help='rotation to apply (90, 180, 270)')
    parser.add_argument('--roi', metavar=('x0', 'y0', 'x1', 'y1', 'x2', 'y2',
                        'x3', 'y3'), type=int, nargs=8, 
                        help='corners of the ROI around the digits')
    parser.add_argument('--digwidth', type=int, metavar='W',
                        help='width of digit in pixels')
    parser.add_argument('--segwidth', type=int, metavar='W',
                        help='width of segment in pixels')
    parser.add_argument('--segthresh', type=float, metavar='T', default=0.35,
                        help='threshold value to consider segment filled or not, between 0 and 1')
    parser.add_argument('--store_crop', type=str, metavar='storedir',
                        help='store cropped, normalized image (e.g. for longer-term debugging)')

    parser.add_argument('--minval', type=int, default=0, metavar='N',
                        help='minimum value to accept (e.g. for incrementing counters)')
    parser.add_argument('--maxincrease', type=int, default=None, metavar='N',
                        help='maximum increase to accept (e.g. for incrementing counters)')

    parser.add_argument('--domoticz', type=str, metavar=("protocol","ip","port", "idx"), default=None,
                        nargs=4, help='Push to domoticz: protocol (http/https), ip, port, and meter idx, e.g. "https 127.0.0.1 10443 24')
    parser.add_argument('--influxdb', type=str, metavar=("protocol","ip","port","db", "query"), default=None,
                        nargs=5, help='Push to influxdb: protocol (http/https), \
                        ip, port, database, and query string which will be \
                        appended with the measurement data, e.g. \
                        "https 127.0.0.1 8086 smarthome "energy,type=heat,device=landisgyr value=""')

    parser.add_argument('--calibrate', type=str, metavar='camfile', nargs="?",
                        help='calibrate parameters, either getting camera image directly (if possible), or using the referenced file')

    parser.add_argument('--logfile', type=str, metavar='path',
                        help='log stuff here')
    parser.add_argument('--debug', action='store_true',
                        help='show debug output during prep_proc calibration')

    # Pre-process command-line arguments
    args = parser.parse_args()
    logging.basicConfig(filename=args.logfile, level=logging.INFO, format='%(asctime)s %(message)s')

    if (args.debug):
        print (args)

    # Check if path exists. args.store_crop is either None or a string
    if (args.store_crop and (not os.path.exists(args.store_crop))):
        logging.warn("Store crop path does not exist, disabling.")
        args.store_crop = None

    if (not args.calibrate):
        if (args.roi == None or args.digwidth == None or args.segwidth == None or args.segthresh == None):
            raise ValueError("Cannot run without options roi, digwidth, segwidth, segthresh. Please calibrate first.")
    
    if (args.roi != None):
        args.roi = np.r_[args.roi].astype(int).reshape(-1,2)

    # Run main program, either in calibration mode or in analysis mode
    if (args.calibrate):
        # Unset default for calibration
        args.segthresh = None
        try:
            im_path, img_data = capture_img(method='file')
            opt_str = calibrate_image(im_path, rotate=args.rotate, roi=args.roi, ndigit=args.ndigit, digwidth=args.digwidth, segwidth=args.segwidth, segthresh=args.segthresh, debug=args.debug)
        except:
            opt_str = calibrate_image(args.calibrate, rotate=args.rotate, roi=args.roi, ndigit=args.ndigit, digwidth=args.digwidth, segwidth=args.segwidth, segthresh=args.segthresh, debug=args.debug)

        print("Calibrated args: {}".format(opt_str))
    else:
        try:
            im_path, img_data = capture_img(method='data')
        except:
            logging.error("Could not acquire image, aborting")
            return

        img_norm, img_thresh = preproc_img(im_path, img_data, roi=args.roi, rotate=args.rotate, store_crop=args.store_crop, debug=args.debug)
        lcd_digit_levels = read_digits(img_thresh, ndigit=args.ndigit, digwidth=args.digwidth, segwidth=args.segwidth, debug=args.debug)
        
        lcd_value, lcd_probability = calc_value(lcd_digit_levels, segthresh=args.segthresh, minval=args.minval, maxinc=args.maxincrease)

        print("Found: {}, {}".format(lcd_value, lcd_probability))
        if (args.domoticz != None):
            domoticz_update(lcd_value, prot=args.domoticz[0], ip=args.domoticz[1], port=args.domoticz[2], m_idx=args.domoticz[3])
        if (args.influxdb != None):
            influxdb_update(lcd_value, prot=args.influxdb[0], ip=args.influxdb[1], port=args.influxdb[2], db=args.influxdb[3], query=args.influxdb[4])


if __name__ == "__main__":
    main()
    exit()