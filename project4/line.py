import numpy as np
import cv2

class LineWindow:
    def __init__(self, nonzero, window_height, margin):
        # Create empty lists to receive left and right lane pixel indices
        self.lane_indexes = []

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
        fit = np.polyfit(py, px, 2)

        line = Line(lane_inds, fit, self.windows)
        return line

class Line:
    def __init__(self, lane_indexes, fit, windows):
        self.lane_indexes = lane_indexes
        self.fit = fit
        self.windows = windows

    def PlotFit(self, canvas, image_size):
        # Generate x and y values for plotting
        y = image_size[1]
        ploty = np.linspace(0, y-1, y)
        fit = self.fit
        fitx = fit[0] * ploty**2 + fit[1] * ploty + fit[2]

        #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        canvas.plot(fitx, ploty, color='yellow')

    def DrawWindows(self, image, color=(0,255,0)):
        # Draw the windows on the visualization image
        for window in self.windows:
            cv2.rectangle(image, (window[0], window[1]), (window[2], window[3]), color, 2)

class LineLocator:

    def __init__(self, image_size):
        self.x = image_size[0]
        self.y = image_size[1]
        self.midpoint = int(self.x/2)

    # nwindows - choose the number of sliding windows
    # margin - the width of the windows +/- margin
    # minpix - minimum number of pixels found to recener window
    def Locate(self, image, nwindows = 9, margin=100, minpix=50, show=False):
        # NOTE np encoding for images (y,x)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image.nonzero()

        window_height = int(self.y / nwindows)
        left = LineWindow(nonzero, window_height, margin)
        right = LineWindow(nonzero, window_height, margin)

        # finding base locations for left and right lines
        lower = image[self.midpoint:,:]
        base = np.sum(lower, axis=0)
        left.SetBase(np.argmax(base[:self.midpoint]))
        right.SetBase(np.argmax(base[self.midpoint:]) + self.midpoint)

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

        return left_line, right_line
