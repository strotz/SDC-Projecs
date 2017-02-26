import camera
import cv2
import gradients
import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import line

from moviepy.editor import VideoFileClip

def show_images(original, processed):
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(original)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(processed)
    ax2.set_title('Processed', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    pylab.show()

def save_image(image, name):
    plt.imsave('out/' + name, image)

def binary_to_color(binary):
    color_binary = np.dstack((binary, binary, binary))
    return color_binary * 255



#
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

#### source of the calibration
dataf='/Users/Shared/SDC/CarND-Advanced-Lane-Lines/'
calibration_set = camera.CameraCalibrationSet(dataf + 'camera_cal/c*.jpg')

cam = camera.Camera()
cam.LoadCalibrationSet(calibration_set)

#### source of the images
# test = dataf + 'test_images/test1.jpg'
# img = mpimg.imread(test)
videoin = dataf + 'project_video.mp4'
clip = VideoFileClip(videoin, audio=False)
img = clip.make_frame(0)
img_size = (img.shape[1], img.shape[0])
#save_image(img, '01_first_image_from_clip.png')

cam.CalibrateFor(img_size)
img = cam.Undistort(img)
#save_image(img, '02_undistorted.png')

builder = camera.ViewPointBuilder.New()
builder.SetHorizonLine(0.65)
builder.SetBottomLine(0.96)
builder.SetNearView(0.8)
builder.SetFarView(0.16)
view = builder.BuildView(img_size)
img_bv = view.MakeBirdView(img)
#save_image(img_bv, '03_bird_view.png')

# apply filters
ms = gradients.magnitude_sobel(img, thresholds=(50,255))
cs = gradients.hsv_mix(img, 'S', thresholds=(100,255))
binary = np.zeros_like(img[:,:,0], dtype=np.uint8)
binary[(ms==1)|(cs==1)]=1
#save_image(binary_to_color(binary), '04_filtered_by_sobel_and_color.png')

binary_bv = view.MakeBirdView(binary)
#save_image(binary_to_color(binary_bv), '05_filtered_bird_view.png')

locator = line.LaneLocator(img_size)
lane = locator.Locate(binary_bv) # search using sliding windows

# TODO: move to function
#out_img = binary_to_color(binary_bv)
#lane.l.DrawSearchArea(out_img)
#lane.l.PlotFit(out_img)
#lane.r.DrawSearchArea(out_img)
#lane.r.PlotFit(out_img)
#save_image(out_img, '06_sliding_windows_and_fitted_polynom.png')

#lr = lane.l.CalculateRadius(xm_per_pix, ym_per_pix)
#rr = lane.r.CalculateRadius(xm_per_pix, ym_per_pix)

# TODO: add low pass filter for fit
# TODO: add convolutions

out = np.zeros_like(img_bv[:,:,:], dtype=np.uint8)
lane.Draw(out)
reverted = view.RevertBirdView(out)
result = cv2.addWeighted(img, 1, reverted, 0.3, 0)
#save_image(result, '07_lane_applied_to_original.png')

def process_clip_frame(image):
    global lane
    img = cam.Undistort(image)

    # apply filters
    ms = gradients.magnitude_sobel(img, thresholds=(50,255))
    cs = gradients.hsv_mix(img, 'S', thresholds=(100,255))
    binary = np.zeros_like(img[:,:,0], dtype=np.uint8)
    binary[(ms==1)|(cs==1)]=1
    binary_bv = view.MakeBirdView(binary)

    lane = locator.Adjust(binary_bv, lane) # search using previous fit

    # TODO: move to function
    #out_img = binary_to_color(binary_bv)
    #lane.l.DrawSearchArea(out_img)
    #lane.l.PlotFit(out_img)
    #lane.r.DrawSearchArea(out_img)
    #lane.r.PlotFit(out_img)
    #save_image(out_img, '08_fitting_adjusted.png')

    out = np.zeros_like(img_bv[:,:,:], dtype=np.uint8)
    lane.Draw(out)
    reverted = view.RevertBirdView(out)
    result = cv2.addWeighted(img, 1, reverted, 0.3, 0)

    # Write some Text
    #font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(img,'Hello World!',(10,500), font, 1,(255,255,255),2)
    return result

img2 = clip.make_frame(1)
result2 = process_clip_frame(img2)
#save_image(result2, '09_second_frame_processed.png')

lane_found_clip = clip.fl_image(process_clip_frame)
lane_found_clip.write_videofile('out/lane_detected.mp4', audio=False)
