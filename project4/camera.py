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
        pack["img_size"] = self.image_size
        pack["mtx"] = self.mtx
        pack["dist"] = self.dist
        pickle.dump(pack, open(fname, "wb"))

    def LoadCalibration(self, fname):
        with open(fname, 'rb') as f:
            pack = pickle.load(f)
            self.image_size = pack["img_size"]
            self.mtx = pack["mtx"]
            self.dist = pack["dist"]
             

class ViewPointBuilder:
    def __init__(self):
        self.horizon = 0
        self.bottom = 1.0
        self.near = 1.0
        self.far = 1.0

    @staticmethod
    def New():
        return ViewPointBuilder()

    def SetHorizonLine(self, horizon):
        self.horizon = horizon

    def SetBottomLine(self, bottom):
        self.bottom = bottom

    def SetNearView(self, near):
        self.near = near

    def SetFarView(self, far):
        self.far = far

    def BuildView(self, image_size):
        # y
        y=image_size[1]
        horizon_px = y * self.horizon
        bottom_px = y * self.bottom

        # x
        x = image_size[0]
        far_left = x * (0.5 - self.far / 2)
        far_right = x * (0.5 + self.far / 2)
        near_left = x * (0.5 - self.near / 2)
        near_right = x * (0.5 + self.near / 2)
        ox = x * (0.5 - self.near / 2)

        src = np.float32([[near_left,bottom_px],[far_left,horizon_px],[far_right,horizon_px],[near_right,bottom_px]])
        dst = np.float32([[ox, y],[ox, 0],[x-ox, 0],[x-ox,y]])

        return ViewPoint(src, dst)

class ViewPoint:
    def __init__(self, src, dest):
        self.M = cv2.getPerspectiveTransform(src, dest)
        self.Minverse = cv2.getPerspectiveTransform(dest, src)

    def MakeBirdView(self, image):
        img_size = (image.shape[1], image.shape[0])
        return cv2.warpPerspective(image, self.M, img_size)

    def RevertBirdView(self, image):
        img_size = (image.shape[1], image.shape[0])
        return cv2.warpPerspective(image, self.Minverse, img_size)
