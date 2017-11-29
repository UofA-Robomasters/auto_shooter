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


def color_judge(image, relative_thresh=30, color_thresh=100, pixel_thresh=60, pixel_ratio=2):
    """
    Decide if this image contains red LEDs or blue LEDs or neither
    :param image: cropped image containing the armor bard and the LEDs
    :param relative_thresh: The threshold for r/b channel - g channel
    :param color_thresh: The threshold for r/b channel
    :param pixel_thresh: The threshold for the number of pixel pass
    :param pixel_ratio: The threshold for the number of r/b pixels divided by the number of b/r pixels
    :return: one of three strings. "r" - red team, "b" - blue team, "None" - neither
    """
    # seperate color channels
    r_channel = image[:, :, 0]
    g_channel = image[:, :, 1]
    b_channel = image[:, :, 2]

    # Red team?
    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel > g_channel) & (r_channel - g_channel > relative_thresh) & (r_channel > b_channel) & (r_channel - b_channel > relative_thresh) & (r_channel > color_thresh)] = 1

    # Blue team?
    b_binary = np.zeros_like(r_channel)
    b_binary[(b_channel > g_channel) & (b_channel - g_channel > relative_thresh) & (b_channel > r_channel) & (b_channel - r_channel > relative_thresh) & (b_channel > color_thresh)] = 1

    # count the pixels passed
    r_pixel = r_binary.sum()
    b_pixel = b_binary.sum()

    # decide which team does it belongs to
    if r_pixel > pixel_thresh and (b_pixel == 0 or r_pixel / b_pixel > pixel_ratio):
        return "r"
    elif b_pixel > pixel_thresh and (r_pixel == 0 or b_pixel / r_pixel > pixel_ratio):
        return "b"
    else:
        return "None"


def draw_circle(image, x, y, size, color):
    if color == "r":
        cv2.circle(image, (x, y), int(size / 2), (255, 0, 0), -1)
    elif color == "b":
        cv2.circle(image, (x, y), int(size / 2), (0, 0, 255), -1)
    else:
        cv2.circle(image, (x, y), int(size / 2), (255, 255, 0), -1)
    return image


def configure_params(filterByArea, minArea, maxArea, filterbyCircularity, minCircularity, filterByConvexity, minConvexity, filterByInertia, minInertiaRatio, maxInertiaRatio,):
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 255

    # Filter by Area.
    params.filterByArea = filterByArea
    params.minArea = minArea
    params.maxArea = maxArea

    # Filter by Circularity
    params.filterByCircularity = filterbyCircularity
    params.minCircularity = minCircularity

    # Filter by Convexity
    params.filterByConvexity = filterByConvexity
    params.minConvexity = minConvexity

    # Filter by Inertia
    params.filterByInertia = filterByInertia
    params.minInertiaRatio = minInertiaRatio
    params.maxInertiaRatio = maxInertiaRatio

    params.filterByColor = False
    params.blobColor = 255

    return params


def bound_image_x(x):
    return int(max(0, min(x, 1280)))


def bound_image_y(y):
    return int(max(0, min(y, 720)))


def bound_masked_x(x):
    return int(max(0, min(x, w_)))


def bound_masked_y(y):
    return int(max(0, min(y, h_)))


class Armour():
    def __init__(self, x, y, size, color):
        # coordinate
        self.x = x
        self.y = y
        # size
        self.size = size
        # red or blue
        self.color = color
        # displacement tolerance
        self.x_thresh = 10
        self.y_thresh = 5

    def is_same(self, x, y, color):
        if color == self.color and abs(x - self.prev_x) < self.x_thresh and abs(y - self.prev_y) < self.y_thresh:
            return True
        else:
            return False

    def update(self, x, y, size, color):
        self.x = x
        self.y = y
        self.size = size
        self.color = color


def process_image(image):
    global armour_list, global_search, frame_cnt
    frame_cnt += 1
    # resize input images to given given dimensions
    image = cv2.resize(image, (1280, 720))
    # crop out the region of interest
    # masked_image = region_of_interest(image, vertices)
#     plt.imshow(image)
    masked_image = image[y_:y_+h_, x_:x_+w_]
#     print(x_, y_, w_, h_)
#     plt.imshow(masked_image)
    # remove some noise
    masked_image = cv2.medianBlur(masked_image, 5)
    # global color thresholding
    gray_binary = gray_threshold(masked_image, thresh=(190, 255))
    # local gray thresholding
    gray_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    threshold = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 111, -60)
    # combined two binary images
    combined = np.zeros_like(gray_image)
    combined[((gray_binary == 1) | (threshold == 255))] = 255

    # armour detection
    this_armour_list = []
    im_with_keypoints = image

    if armour_list == [] or frame_cnt % 30 == 0:
        global_search = True
    # tracking
    if not global_search:
        armour_found = 0
        total_armour = len(armour_list)
        for armour in armour_list:
            # crop the local image out
            half_width = int(armour.size + 20)
            half_height = int((armour.size + 20) // 2)
            y1 = bound_masked_y(armour.y - half_height - y_)
            y2 = bound_masked_y(armour.y + half_height - y_)
            x1 = bound_masked_x(armour.x - half_width - x_)
            x2 = bound_masked_x(armour.x + half_width - x_)
            crop_combined = combined[y1:y2, x1:x2]
            y1 = bound_image_y(armour.y - half_height)
            y2 = bound_image_y(armour.y + half_height)
            x1 = bound_image_x(armour.x - half_width)
            x2 = bound_image_x(armour.x + half_width)
            crop_image = image[y1:y2, x1:x2]
            keypoints = easier_detector.detect(crop_combined)
            # find the biggest keypoint (highest possibility)
            largest_keypoint = None
            largest_size = 0
            for keypoint in keypoints:
                if keypoint.size > largest_size:
                    largest_size = keypoint.size
                    largest_keypoint = keypoint
            # lower threshold
            judge_result = color_judge(crop_image, 20, 80, 50, 2)
            if largest_keypoint is not None and judge_result == armour.color:
                x = int(largest_keypoint.pt[0])
                y = int(largest_keypoint.pt[1])
                this_armour_list.append(Armour(x1 + x, y1 + y, largest_keypoint.size, judge_result))
                armour_found += 1
        # if armour_found != total_armour:
        #     global_search = True
        else:
            cv2.putText(im_with_keypoints, "Tracking", (500, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
    # global search
    if global_search:
        # blob dectector for circle
        keypoints = bolb_detector.detect(combined)
        # draw the detected circles
        cv2.putText(im_with_keypoints, "Searching", (500, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        for keypoint in keypoints:
            x = int(keypoint.pt[0] + x_)
            y = int(keypoint.pt[1] + y_)
            half_width = int(keypoint.size + 10)
            half_height = (keypoint.size + 10) // 2
            y1 = bound_image_y(y - half_height)
            y2 = bound_image_y(y + half_height)
            x1 = bound_image_x(x - half_width)
            x2 = bound_image_x(x + half_width)
            crop_image = image[y1:y2, x1:x2]
            # color judge
            judge_result = color_judge(crop_image, 30, 100, 60, 2)
            this_armour_list.append(Armour(x, y, keypoint.size, judge_result))
        global_search = False
    # update the armour list
    armour_list = this_armour_list
    # draw colored circles
    for armour in armour_list:
        im_with_keypoints = draw_circle(im_with_keypoints, armour.x, armour.y, armour.size, armour.color)
    # draw the region of interest
    cv2.rectangle(im_with_keypoints, (x_, y_), (x_ + w_, y_ + h_), (0, 255, 0), 5)
    return im_with_keypoints


if __name__ == "__main__":
    start_time = time.time()
    blob_params = configure_params(True, 200, 5000, True, 0.7, True, 0.7, True, 0.1, 1.0)
    bolb_detector = cv2.SimpleBlobDetector_create(blob_params)
    easier_params = configure_params(True, 150, 5000, True, 0.5, True, 0.5, True, 0.1, 1.0)
    easier_detector = cv2.SimpleBlobDetector_create(easier_params)
    armour_list = []
    global_search = True
    x_ = 200
    y_ = 100
    w_ = 830
    h_ = 380
    frame_cnt = 0
    # vertices = np.array([[(x_, y_ + h_), (x_, y_), (x_ + w_, y_), (x_ + w_, y_ + h_)]], dtype=np.int32)
    video_output1 = 'test_output.mp4'
    video_input1 = VideoFileClip('videos/test_video.mpeg')#.subclip(1, 2)
    processed_video = video_input1.fl_image(process_image)
    processed_video.write_videofile(video_output1, audio=False)
    print("time it takes:", time.time() - start_time)
