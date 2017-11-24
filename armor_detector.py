import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import time


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def rgb_select(image, thresh=(0, 255), color='b'):
    if color == 'b':
        color_channel = image[:, :, 2]
    elif color == 'r':
        color_channel = image[:, :, 0]

    binary_output = np.zeros_like(color_channel)
    binary_output[(color_channel > thresh[0]) & (color_channel <= thresh[1])] = 1
    return binary_output


def hsv_select(image, thresh=(0, 255), color='s'):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if color == 'h':
        color_channel = hls[:, :, 0]
    elif color == 's':
        color_channel = hls[:, :, 1]
    elif color == 'v':
        color_channel = hls[:, :, 2]

    binary_output = np.zeros_like(color_channel)
    binary_output[(color_channel > thresh[0]) & (color_channel <= thresh[1])] = 1
    return binary_output


def gray_threshold(image, thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary_output = np.zeros_like(gray)
    binary_output[(gray > thresh[0]) & (gray <= thresh[1])] = 1
    return binary_output

def process_image(image):
    # resize input images to given given dimensions
    image = cv2.resize(image, (1280, 720))

    # crop out the region of interest
    x_ = 200
    y_ = 100
    w_ = 850
    h_ = 400
    vertices = np.array([[(x_, y_ + h_), (x_, y_), (x_ + w_, y_), (x_ + w_, y_ + h_)]], dtype=np.int32)
    masked_image = region_of_interest(image, vertices)

    # remove some noise
    blur_image = cv2.medianBlur(masked_image, 5)

    # color threshold
    gray_binary = gray_threshold(blur_image, thresh=(190, 255))
    # local threshold
    gray_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    threshold = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 111, -60)

    # combined two binary images
    combined = np.zeros_like(gray_binary)
    combined[((gray_binary == 1) | (threshold == 255))] = 255

    # blob dectector for circle
    detector = cv2.SimpleBlobDetector_create(blob_params)
    keypoints = detector.detect(combined)

    # draw the detected circles
    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (255, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    for keypoint in keypoints:
        cv2.circle(im_with_keypoints, (int(keypoint.pt[0]), int(keypoint.pt[1])), 7, (255, 255, 0), -1)

    # draw the region of interest
    cv2.rectangle(im_with_keypoints, (x_, y_), (x_ + w_, y_ + h_), (0, 255, 0), 5)

    return im_with_keypoints

def configure_params():
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 0;
    params.maxThreshold = 255;

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 200
    params.maxArea = 5000

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.7

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.5

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    params.maxInertiaRatio = 1.0

    params.filterByColor = False
    params.blobColor = 255

    return params

if __name__ == "__main__":
    start_time = time.time()
    blob_params = configure_params()
    video_output1 = 'test_output.mp4'
    video_input1 = VideoFileClip('videos/test_video.mpeg')#.subclip(1, 2)
    processed_video = video_input1.fl_image(process_image)
    processed_video.write_videofile(video_output1, audio=False)
    print("time it takes:", time.time() - start_time)
