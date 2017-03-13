import pylab
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage as ndimage

def show_images(original, processed):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(original)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(processed)
    ax2.set_title('Processed', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    pylab.show()

def show_heat(draw_img, heatmap):
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(draw_img)
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()
    pylab.show()

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy

def draw_labeled_bboxes(img, labels):
    objects = ndimage.find_objects(labels[0])
    for box in objects:
        y = box[0]
        x = box[1]
        cv2.rectangle(img, (x.start, y.start), (x.stop, y.stop), (0,0,255), 6)
    return img

def binary_to_color(binary):
    color_binary = np.dstack((binary, binary, binary))
    return color_binary * 255

demo = True
def save_image(image, name):
    if demo == False:
        plt.imshow(image)
        pylab.show()
        return

    plt.imsave('out/' + name, image)

# Function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(32, 32), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None: x_start_stop[0]=0
    if x_start_stop[1] == None: x_start_stop[1]=img.shape[1]
    if y_start_stop[0] == None: y_start_stop[0]=0
    if y_start_stop[1] == None: y_start_stop[1]=img.shape[0]

    # Compute the span of the region to be searched
    x_span = x_start_stop[1] - x_start_stop[0]
    y_span = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    x_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    y_step = np.int(xy_window[1] * (1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    x_nwindows = np.int((x_span - (xy_window[0] * xy_overlap[0]))/x_step)
    y_nwindows = np.int((y_span - (xy_window[1] * xy_overlap[1]))/y_step)

    # Initialize a list to append window positions to
    window_list = []
    for x in range(x_nwindows):
        for y in range(y_nwindows):
            # Calculate each window position
            startx = x * x_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = y * y_step + y_start_stop[0]
            endy = starty + xy_window[1]
            window_list.append(((startx, starty), (endx, endy)))

    return window_list

def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap

def add_heat_value(heatmap, bbox_list, values):
    i = 0
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += values[i]
        i += 1
    return heatmap

def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0.0
    return heatmap

def split_image(image, boxes, resize_to=None):
    parts_list = []
    for box in boxes:
        part = image[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        if resize_to != None and part.shape[0] != resize_to[1] and part.shape[0] != resize_to[0]:
            part = cv2.resize(part, resize_to)
        parts_list.append(part)

    return np.asarray(parts_list)
