import camera
import cv2
import gradients
import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import line

from moviepy.editor import VideoFileClip

demo = False

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
    if demo == False:
        return
    plt.imsave('out/' + name, image)

def binary_to_color(binary):
    color_binary = np.dstack((binary, binary, binary))
    return color_binary * 255

class ImageProcessing:
    def __init__(self, img_size, calibration_set_pattern):
        self.img_size = img_size

        calibration_set = camera.CameraCalibrationSet(calibration_set_pattern)
        self.cam = camera.Camera()
        self.cam.LoadCalibrationSet(calibration_set)
        self.cam.CalibrateFor(self.img_size)

        builder = camera.ViewPointBuilder.New()
        builder.SetHorizonLine(0.65)
        builder.SetBottomLine(0.96)
        builder.SetNearView(0.8)
        builder.SetFarView(0.16)
        self.view = builder.BuildView(img_size)

        self.locator = line.LaneLocator(img_size)

        self.last_lane = None

    def Filter(self, image):
        # apply filters
        ms = gradients.magnitude_sobel(image, thresholds=(50,255))
        cs = gradients.hsv_mix(image, 'S', thresholds=(100,255))
        binary = np.zeros_like(image[:,:,0], dtype=np.uint8)
        binary[(ms==1)|(cs==1)]=1
        return binary

    def ApplyLane(self, original, lane):
        out = np.zeros_like(original[:,:,:], dtype=np.uint8)
        lane.Draw(out)
        reverted = self.view.RevertBirdView(out)
        return cv2.addWeighted(original, 1, reverted, 0.3, 0)


    def UseSlidingWindow(self, original):
        img = self.cam.Undistort(original)
        binary = self.Filter(img)
        binary_bv = self.view.MakeBirdView(binary)
        lane = self.locator.Locate(binary_bv) # search using sliding windows
        result = self.ApplyLane(original, lane)
        self.last_lane = lane
        return result

    def UseAdjuster(self, original):
        img = self.cam.Undistort(original)
        binary = self.Filter(img)
        binary_bv = self.view.MakeBirdView(binary)
        lane = self.locator.Adjust(binary_bv, self.last_lane) # search using previous fit
        result = self.ApplyLane(original, lane)
        self.last_lane = lane
        return result

    def UseSmartLocate(self, original):
        img = self.cam.Undistort(original)
        binary = self.Filter(img)
        binary_bv = self.view.MakeBirdView(binary)
        lane = self.locator.SmartLocate(binary_bv, self.last_lane) # search using previous fit or sliding window
        result = self.ApplyLane(original, lane)
        self.last_lane = lane
        return result

    #
    # step by step process to capture images
    #
    def Demo(self, original):
        save_image(original, '01_first_image_from_clip.png')
        img = self.cam.Undistort(original)
        save_image(img, '02_undistorted.png')
        img_bv = self.view.MakeBirdView(img)
        save_image(img_bv, '03_bird_view.png')
        binary = self.Filter(img)
        save_image(binary_to_color(binary), '04_filtered_by_sobel_and_color.png')
        binary_bv = self.view.MakeBirdView(binary)
        save_image(binary_to_color(binary_bv), '05_filtered_bird_view.png')

        lane = self.locator.Locate(binary_bv) # search using sliding windows
        out_img = lane.DrawSearch(binary_bv)
        save_image(out_img, '06_sliding_windows_and_fitted_polynom.png')

        result = self.ApplyLane(original, lane)
        out_img = lane.DrawSearch(binary_bv)
        save_image(result, '07_lane_applied_to_original.png')

        lane = self.locator.Adjust(binary_bv, lane) # search using previous fit
        save_image(out_img, '08_fitting_adjusted.png')

#
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

#### source of data
dataf='/Users/Shared/SDC/CarND-Advanced-Lane-Lines/'
calibration_set_pattern = dataf + 'camera_cal/c*.jpg'

#### source of the images
#test = dataf + 'test_images/test4.jpg'
#original = mpimg.imread(test)
videoin = dataf + 'project_video.mp4'
clip = VideoFileClip(videoin, audio=False)
original = clip.make_frame(700)

img_size = (original.shape[1], original.shape[0])

processing = ImageProcessing(img_size, calibration_set_pattern)
result = processing.UseSmartLocate(original)
# open show_images(original, result)

#lr = lane.l.CalculateRadius(xm_per_pix, ym_per_pix)
#rr = lane.r.CalculateRadius(xm_per_pix, ym_per_pix)

# TODO: add low pass filter for fit
# TODO: add convolutions

def process_clip_frame(image):
    # Write some Text
    global processing
    #font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(img,'Hello World!',(10,500), font, 1,(255,255,255),2)
    return processing.UseSmartLocate(image)

# TODO: move to function
lane_found_clip = clip.fl_image(process_clip_frame)
lane_found_clip.write_videofile('out/lane_detected.mp4', audio=False)
