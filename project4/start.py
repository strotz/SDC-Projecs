# import Camera
import camera
import cv2

calibration_set = camera.CameraCalibrationSet('/Users/Shared/SDC/CarND-Advanced-Lane-Lines/camera_cal/c*.jpg')

cam = camera.Camera()
cam.LoadCalibrationSet(calibration_set)

img = cv2.imread(calibration_set.ImageAt(0))
img_size = (img.shape[1], img.shape[0])

cam.CalibrateFor(img_size)
img = cam.Undistort(img)

cv2.imshow('img', img)
cv2.waitKey(5000)
# cv2.destroyAllWindows()
