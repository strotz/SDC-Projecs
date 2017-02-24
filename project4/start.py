import camera
import cv2
import gradients
import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

####
calibration_set = camera.CameraCalibrationSet('/Users/Shared/SDC/CarND-Advanced-Lane-Lines/camera_cal/c*.jpg')

cam = camera.Camera()
cam.LoadCalibrationSet(calibration_set)

####
test = '/Users/Shared/SDC/CarND-Advanced-Lane-Lines/test_images/test1.jpg'
img = mpimg.imread(test)
img_size = (img.shape[1], img.shape[0])

cam.CalibrateFor(img_size)
img = cam.Undistort(img)

builder = camera.ViewPointBuilder.New()
builder.SetHorizonLine(0.62)
builder.SetBottomLine(0.95)
builder.SetNearView(0.8)
builder.SetFarView(0.11)
view = builder.BuildView(img_size)

# apply filters
ms = gradients.magnitude_sobel(img, thresholds=(50,255))
cs = gradients.hsv_mix(img, 'S', thresholds=(100,255))

binary = np.zeros_like(img[:,:,0], dtype=np.uint8)
binary[(ms==1)|(cs==1)]=1

img_bv = view.MakeBirdView(img)
binary_bv = view.MakeBirdView(binary)

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img_bv)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(binary_bv, cmap='gray', vmin=0, vmax=1)
ax2.set_title('Thresholded', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
pylab.show()
