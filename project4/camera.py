import glob
import cv2
import numpy as np
import pickle

class CameraCalibrationSet:
    def __init__(self, source):
        self.pointsX=9
        self.pointsY=6
        self.images = glob.glob(source) # '/Users/Shared/SDC/CarND-Advanced-Lane-Lines/camera_cal/c*.jpg'

    def Enumerate(self):
        return enumerate(self.images)

    def ImageAt(self, idx):
        return self.images[idx]

class Camera:
    def LoadCalibrationSet(self, calibration_set, show=False):
        p_x = calibration_set.pointsX
        p_y = calibration_set.pointsY

        objp = np.zeros((p_y * p_x, 3), np.float32)

        # TODO: what this is
        objp[:,:2] = np.mgrid[0:p_x, 0:p_y].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d points in real world space
        self.imgpoints = [] # 2d points in image plane.

        # Step through the list and search for chessboard corners
        for idx, fname in calibration_set.Enumerate():
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (p_x, p_y), None)

            # If found, add object points, image points
            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)

                if show == True:
                    # Draw and display the corners
                    cv2.drawChessboardCorners(img, (p_x, p_y), corners, ret)
                    cv2.imshow('img', img)
                    cv2.waitKey(500)

        cv2.destroyAllWindows()

    def CalibrateFor(self, image_size):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, image_size, None, None)
        self.image_size = image_size
        self.mtx = mtx
        self.dist = dist
        return ret

    def Undistort(self, image):
        if image.shape[1] != self.image_size[0] or image.shape[0] != self.image_size[1]:
            raise ValueError('wrong calibration')
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

    def SaveCalibraton(self, fname):
        pack = {}
        pack["mtx"] = self.mtx
        pack["dist"] = self.dist
        pickle.dump(pack, open(fname, "wb"))
