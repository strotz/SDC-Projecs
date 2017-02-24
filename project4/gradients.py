import cv2
import numpy as np

def mask(img, thresholds):
    threshold_min = thresholds[0]
    threshold_max = thresholds[1]
    binary = np.zeros_like(img, dtype=np.uint8)
    binary[(img >= threshold_min) & (img <= threshold_max)] = 1
    return binary

def absolute_sobel(img, orient='x', sobel_kernel=3, thresholds=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel) if orient=='x' else cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    return mask(scaled_sobel, thresholds)

def magnitude_sobel(img, sobel_kernel=3, thresholds=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    return mask(scaled_sobel, thresholds)

def direction_sobel(img, sobel_kernel=3, thresholds=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    scaled_sobel = np.arctan2(abs_sobely, abs_sobelx)
    return mask(scaled_sobel, thresholds)

def hls_mix(img, channel, thresholds=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    color = {
    'H': hls[:,:,0],
    'L': hls[:,:,1],
    'S': hls[:,:,2]
    }[channel]
    return mask(color, thresholds)

def hsv_mix(img, channel, thresholds=(0, 255)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    color = {
    'H': hsv[:,:,0],
    'S': hsv[:,:,1],
    'V': hsv[:,:,2]
    }[channel]
    return mask(color, thresholds)
