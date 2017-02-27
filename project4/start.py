import camera
import cv2
import gradients
import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import line
from moviepy.editor import VideoFileClip
import lpf
import os.path

demo = False
# TODO: add convolutions

font = cv2.FONT_HERSHEY_SIMPLEX

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30.0 / 720.0 # meters per pixel in y dimension
xm_per_pix = 3.7 / 700.0 # meters per pixel in x dimension

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
        plt.imshow(image)
        pylab.show()
        return
    plt.imsave('out/' + name, image)

def binary_to_color(binary):
    color_binary = np.dstack((binary, binary, binary))
    return color_binary * 255

class ImageProcessing:
    def __init__(self, img_size, calibration_set_pattern):
        self.img_size = img_size
          
        packfile = './calibration.pk'
        if os.path.isfile(packfile):
            print('loading calibration')
            self.cam = camera.Camera()
            self.cam.LoadCalibration(packfile)
        else:
            calibration_set = camera.CameraCalibrationSet(calibration_set_pattern)
            self.cam = camera.Camera()
            self.cam.LoadCalibrationSet(calibration_set)
            self.cam.CalibrateFor(self.img_size)
            self.cam.SaveCalibraton(packfile)

        builder = camera.ViewPointBuilder.New()
        builder.SetHorizonLine(0.65)
        builder.SetBottomLine(0.96)
        builder.SetNearView(0.8)
        builder.SetFarView(0.15)
        self.view = builder.BuildView(img_size)

        self.locator = line.LaneLocator(img_size)

        self.last_lane = None

        self.ring = lpf.Smoother(0.4)


    def Filter(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        yellow_low  = np.array([ 10, 80, 100])
        yellow_high = np.array([ 40, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_low, yellow_high)

        white_low  = np.array([  0, 0, 220], dtype=np.uint8)
        white_high = np.array([ 180, 255, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hsv, white_low, white_high)

        # sobel
        sobel_kernel=3
        gray = hsv[:,:,2]

        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

        binary = np.zeros_like(image[:,:,0], dtype=np.uint8) 

        threshold_min = 5
        threshold_max = 255
        binary[(scaled_sobel >= threshold_min)&(scaled_sobel <= threshold_max)&((white_mask!=0)|(yellow_mask!=0))] = 1

        return binary

    def ApplyLane(self, original, lane):
        out = np.zeros_like(original[:,:,:], dtype=np.uint8)
        lane.Draw(out)
        reverted = self.view.RevertBirdView(out)
        result = cv2.addWeighted(original, 1, reverted, 0.3, 0)

        img_size = (original.shape[1], original.shape[0])
        lr, rr, offset = lane.CalaculateRadiuses(img_size, xm_per_pix, ym_per_pix)
        text = 'left: {:6d}m, right: {:6d}m, offset: {:4.2f}m'.format(int(lr), int(rr), offset)
        cv2.putText(result, text,(10,100), font, 1, (255,255,255), 2)

        return result

    def UseSmartLocate(self, original):
        img = self.cam.Undistort(original)
        binary = self.Filter(img)
        binary_bv = self.view.MakeBirdView(binary)
        lane = self.locator.SmartLocate(binary_bv, self.last_lane) # search using previous fit or sliding window
        lane = self.ring.ApplyLPF(lane)
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

        #h = self.locator.BaseCalculation(binary_bv)
        #plt.plot(h)
        #pylab.show()

        lane = self.locator.Locate(binary_bv) # search using sliding windows
        out_img = lane.DrawSearch(binary_bv)
        save_image(out_img, '06_sliding_windows_and_fitted_polynom.png')

        result = self.ApplyLane(original, lane)
        out_img = lane.DrawSearch(binary_bv)
        save_image(result, '07_lane_applied_to_original.png')

        lane = self.locator.Adjust(binary_bv, lane) # search using previous fit
        save_image(out_img, '08_fitting_adjusted.png')
        self.last_lane = lane
        return result

def DemoCalibration(calibration_set_pattern):
    calibration_set = camera.CameraCalibrationSet(calibration_set_pattern)
    cam = camera.Camera()
    cam.LoadCalibrationSet(calibration_set)
    original=original = mpimg.imread(calibration_set.ImageAt(0))
    img_size = (original.shape[1], original.shape[0])
    cam.CalibrateFor(img_size)
    img = cam.Undistort(original)
    save_image(img, '00_undistorted.png')

def ProcessTestImage(calibration_set_pattern):
    test = dataf + 'test_images/test1.jpg'
    original = mpimg.imread(test)
    img_size = (original.shape[1], original.shape[0])
    processing = ImageProcessing(img_size, calibration_set_pattern)
    result = processing.Demo(original)
    show_images(original, result)

def ProcessVideoClip(calibration_set_pattern):
    #### source of the images
    videoin = dataf + 'project_video.mp4'
    clip = VideoFileClip(videoin, audio=False)
    original = clip.make_frame(0)
    img_size = (original.shape[1], original.shape[0])
    processing = ImageProcessing(img_size, calibration_set_pattern)
    def process_clip_frame(image):
        return processing.UseSmartLocate(image)
    result = processing.UseSmartLocate(original)
    lane_found_clip = clip.fl_image(process_clip_frame)
    lane_found_clip.write_videofile('out/lane_detected.mp4', audio=False)


def TroubleshootVideoClip(calibration_set_pattern):
    #### source of the images
    videoin = dataf + 'project_video.mp4'
    clip = VideoFileClip(videoin, audio=False)
    original = clip.make_frame(41.4)
    img_size = (original.shape[1], original.shape[0])
    processing = ImageProcessing(img_size, calibration_set_pattern)
    processing.Demo(original)
    #show_images(original, result)


#### source of data
dataf='../../CarND-Advanced-Lane-Lines/'
calibration_set_pattern = dataf + 'camera_cal/c*.jpg'

ProcessVideoClip(calibration_set_pattern)
