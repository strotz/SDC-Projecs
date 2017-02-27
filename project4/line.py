import numpy as np
import cv2

def binary_to_color(binary):
    color_binary = np.dstack((binary, binary, binary))
    return color_binary * 255

class LineWindow:
    def __init__(self, y, nonzero, window_height, margin):
        # Create empty lists to receive left and right lane pixel indices
        self.lane_indexes = []

        self.y = y
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])

        self.window_height = window_height
        self.margin = margin

    def SetBase(self, x_base):
        self.x_base = x_base
        self.x_current = x_base
        self.lane_indexes = []
        self.windows = []

    def MoveTo(self, y_low):
        self.y_low = y_low
        self.y_high = y_low + self.window_height

        self.x_low = self.x_current - self.margin
        self.x_high = self.x_current + self.margin

        self.windows.append([self.x_low, self.y_low, self.x_high, self.y_high])

    def Search(self, minpix):
        # Identify the nonzero pixels in x and y within the window
        good_inds = ((self.nonzeroy >= self.y_low) & (self.nonzeroy < self.y_high) & (self.nonzerox >= self.x_low) & (self.nonzerox < self.x_high)).nonzero()[0]
        # Append these indices to the lists
        self.lane_indexes.append(good_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_inds) > minpix:
            self.x_current = int(np.mean(self.nonzerox[good_inds]))

    def Fit(self):

        # Concatenate the arrays of indices
        lane_inds = np.concatenate(self.lane_indexes)

        # Extract line pixel positions
        px = self.nonzerox[lane_inds]
        py = self.nonzeroy[lane_inds]

        # Fit a second order polynomial to each
        # fits f(y) because line is often vertical and f(x) can have multiple values
        fit = np.polyfit(py, px, 2)

        return Line(lane_inds, fit, self.y, self.margin, self.windows)

class LineAdjuster:
    def __init__(self, nonzero, margin):
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])

        self.margin = margin

    def Adjust(self, line):
        fit = line.fit
        margin = self.margin
        nonzerox = self.nonzerox
        nonzeroy = self.nonzeroy

        lane_inds = ((nonzerox > (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + fit[2] - margin)) & (nonzerox < (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + fit[2] + margin)))

        # Extract left and right line pixel positions
        px = nonzerox[lane_inds]
        py = nonzeroy[lane_inds]

        # Fit a second order polynomial to each
        fit = np.polyfit(py, px, 2)

        return Line(lane_inds, fit, line.y, margin)

class Line:
    def __init__(self, lane_indexes, fit, y, margin, windows=[]):
        self.lane_indexes = lane_indexes
        self.fit = fit
        self.y = y
        self.margin = margin
        self.windows = windows

    def PlotFit(self, image):
        # Generate x and y values for plotting
        y = image.shape[0]
        ploty = np.linspace(0, y-1, y)
        fit = self.fit
        fitx = fit[0] * ploty**2 + fit[1] * ploty + fit[2]

        points = np.vstack((fitx, ploty)).T
        cv2.polylines(image, np.int32([points]), False, (0,255,255), 3)

    def DrawWindows(self, image, color):
        # Draw the windows on the visualization image
        for window in self.windows:
            cv2.rectangle(image, (window[0], window[1]), (window[2], window[3]), color, 2)

    def DrawPoly(self, image, color, alfa, margin):
        # Generate x and y values for plotting
        y = image.shape[0]
        fit = self.fit

        ploty = np.linspace(0, y-1, y)
        fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]

        # Create an image to draw on and an image to show the selection window
        overlay = np.zeros_like(image)

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        line_window1 = np.array([np.transpose(np.vstack([fitx - margin, ploty]))])
        line_window2 = np.array([np.flipud(np.transpose(np.vstack([fitx + margin, ploty])))])
        line_pts = np.hstack((line_window1, line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(overlay, np.int_([line_pts]), color)
        cv2.addWeighted(image, 1, overlay, alfa, 0, image)

    def DrawSearchArea(self, image, color=(0,255,0)):
        if not self.windows: # empty
            self.DrawPoly(image, color, 0.3, self.margin)
        else:
            self.DrawWindows(image, color)

    def ScaleFit(self, xm_per_pix, ym_per_pix):
        # fit is in pixeles
        A = self.fit[0]
        B = self.fit[1]
        C = self.fit[2]
        x = A * (self.y-1)**2 + B * (self.y-1) + C

        xp_per_m = 1.0 / xm_per_pix
        yp_per_m = 1.0 / ym_per_pix

        # to convert to meters
        # m/p * x => v (m), x = v * (p/m), that gives us new cooficients for fit
        Am = A * yp_per_m**2 / xp_per_m
        Bm = B * yp_per_m / xp_per_m
        Cm = C / xp_per_m
        y = (self.y - 1) / yp_per_m

        return Am, Bm, Cm, y, x

    def CalculateRadius(self, xm_per_pix, ym_per_pix):

        A, B, C, y, x = self.ScaleFit(xm_per_pix, ym_per_pix)
        return ((1 + (2 * A * y + B)**2)**1.5) / np.absolute(2 * A), x

class Lane:
    def __init__(self, l, r):
        self.l = l
        self.r = r

    def Draw(self, image):
        self.l.DrawPoly(image, (0, 0, 255), 1.0, 10)
        self.r.DrawPoly(image, (0, 0, 255), 1.0, 10)

        y = image.shape[0]
        ploty = np.linspace(0, y-1, y)
        left_fitx = self.l.fit[0]*ploty**2 + self.l.fit[1]*ploty + self.l.fit[2]
        right_fitx = self.r.fit[0]*ploty**2 + self.r.fit[1]*ploty + self.r.fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(image, np.int_([pts]), (0,255, 0))

    def DrawSearch(self, image):
        out_img = binary_to_color(image)
        self.l.DrawSearchArea(out_img)
        self.l.PlotFit(out_img)
        self.r.DrawSearchArea(out_img)
        self.r.PlotFit(out_img)
        return out_img

    def CalaculateRadiuses(self, image_size, xm_per_pix, ym_per_pix):
        lr, lx = self.l.CalculateRadius(xm_per_pix, ym_per_pix)
        rr, rx = self.r.CalculateRadius(xm_per_pix, ym_per_pix)
        offset = ((lx + rx) - image_size[0]) / 2 * xm_per_pix
        return lr, rr, offset

class LaneLocator:

    def __init__(self, image_size):
        self.x = image_size[0]
        self.y = image_size[1]
        self.y_midpoint = int(self.y/2)
        self.x_midpoint = int(self.x/2)

    def BaseCalculation(self, image):
        # finding base locations for left and right lines
        lower = image[self.y_midpoint:,:]
        return np.sum(lower, axis=0)

    # nwindows - choose the number of sliding windows
    # margin - the width of the windows +/- margin
    # minpix - minimum number of pixels found to recener window
    def Locate(self, image, nwindows = 9, margin=100, minpix=50):
        # NOTE np encoding for images (y,x)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image.nonzero()

        window_height = int(self.y / nwindows)
        left = LineWindow(self.y, nonzero, window_height, margin)
        right = LineWindow(self.y, nonzero, window_height, margin)

        # finding base locations for left and right lines
        base = self.BaseCalculation(image)
        lbase = np.argmax(base[:self.x_midpoint])
        left.SetBase(lbase)

        rbase = np.argmax(base[self.x_midpoint:]) + self.x_midpoint
        right.SetBase(rbase)

        # Step through the windows one by one
        for window in range(nwindows):

            # Identify window boundaries in x and y (and right and left)
            y_low = self.y - (window + 1) * window_height
            left.MoveTo(y_low)
            right.MoveTo(y_low)

            left.Search(minpix)
            right.Search(minpix)

        left_line = left.Fit()
        right_line = right.Fit()

        return Lane(left_line, right_line)

    def Adjust(self, image, lane, margin=100):
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image.nonzero()

        adjuster = LineAdjuster(nonzero, margin)
        left_line = adjuster.Adjust(lane.l)
        right_line = adjuster.Adjust(lane.r)

        return Lane(left_line, right_line)

    def SmartLocate(self, image, lane, nwindows=9, margin=100, minpix=50):
        if lane is None:
            print('use sliding window')
            return self.Locate(image, nwindows, margin, minpix)

        nonzero = image.nonzero()

        adjuster = LineAdjuster(nonzero, margin)
        left_line = adjuster.Adjust(lane.l)
        right_line = adjuster.Adjust(lane.r)

        if abs(left_line.fit[0] - right_line.fit[0]) > 0.2:
            print('switching to sliding window')
            print(left_line.fit)
            print(right_line.fit)
            return self.Locate(image, nwindows, margin, minpix)

        return self.Locate(image, nwindows, margin, minpix)
