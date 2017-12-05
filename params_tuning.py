from armor_detector import *
import cv2
import glob


def load_image(file_name, width=1280, height=720):
    # load the image
    image = cv2.imread(file_name)
    # resize to specific resolution
    image = cv2.resize(image, (width, height))
    return image


def draw_circle(image, x, y, size, color):
    if color == "r":
        cv2.circle(image, (x, y), int(size / 2), (255, 0, 0), -1)
    elif color == "b":
        cv2.circle(image, (x, y), int(size / 2), (0, 0, 255), -1)
    else:
        cv2.circle(image, (x, y), int(size / 2), (0, 255, 255), -1)
    return image


def image_change(x):
    global image_changed
    image_changed = True


def binary_change(x):
    global binary_changed
    binary_changed = True


def blob_change(x):
    global blob_changed
    blob_changed = True


def judge_change(x):
    global judge_changed
    judge_changed = True


def bound_masked_x(x):
    return int(max(0, min(x, w_)))


def bound_masked_y(y):
    return int(max(0, min(y, h_)))


def color_judge(image, primary_thresh=200, secondary_thresh=80, pixel_thresh=60, pixel_ratio=2):
    # seperate color channels
    r_channel = image[:, :, 0]
    g_channel = image[:, :, 1]
    b_channel = image[:, :, 2]

    # Red team?
    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel > primary_thresh) & (b_channel < secondary_thresh)] = 255

    # Blue team?
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel > primary_thresh) & (r_channel < secondary_thresh)] = 255

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


# def color_judge(image, relative_thresh=30, color_thresh=100, pixel_thresh=60, pixel_ratio=2):
#     """
#     Decide if this image contains red LEDs or blue LEDs or neither
#     :param image: cropped image containing the armor bard and the LEDs
#     :param relative_thresh: The threshold for r/b channel - g channel
#     :param color_thresh: The threshold for r/b channel
#     :param pixel_thresh: The threshold for the number of pixel pass
#     :param pixel_ratio: The threshold for the number of r/b pixels divided by the number of b/r pixels
#     :return: one of three strings. "r" - red team, "b" - blue team, "None" - neither
#     """
#     # seperate color channels
#     r_channel = image[:, :, 0]
#     g_channel = image[:, :, 1]
#     b_channel = image[:, :, 2]
#
#     # Red team?
#     r_binary = np.zeros_like(r_channel)
#     r_binary[(r_channel > g_channel) & (r_channel - g_channel > relative_thresh) & (r_channel > b_channel) & (r_channel - b_channel > relative_thresh) & (r_channel > color_thresh)] = 1
#
#     # Blue team?
#     b_binary = np.zeros_like(r_channel)
#     b_binary[(b_channel > g_channel) & (b_channel - g_channel > relative_thresh) & (b_channel > r_channel) & (b_channel - r_channel > relative_thresh) & (b_channel > color_thresh)] = 1
#
#     # count the pixels passed
#     r_pixel = r_binary.sum()
#     b_pixel = b_binary.sum()
#
#     # decide which team does it belongs to
#     if r_pixel > pixel_thresh and (b_pixel == 0 or r_pixel / b_pixel > pixel_ratio):
#         return "r"
#     elif b_pixel > pixel_thresh and (r_pixel == 0 or b_pixel / r_pixel > pixel_ratio):
#         return "b"
#     else:
#         return "None"


cv2.namedWindow('colored', cv2.WINDOW_NORMAL)
cv2.namedWindow('binary', cv2.WINDOW_NORMAL)
cv2.namedWindow('judged', cv2.WINDOW_NORMAL)

cv2.createTrackbar('blur kernel (2x+1)', 'binary', 2, 10, binary_change)
cv2.createTrackbar('min gray thresh', 'binary', 190, 255, binary_change)
cv2.createTrackbar('max gray thresh', 'binary', 255, 255, binary_change)
cv2.createTrackbar('thresh kernel (2x+1)', 'binary', 55, 100, binary_change)
cv2.createTrackbar('adaptive thresh (-x)', 'binary', 60, 100, binary_change)
cv2.createTrackbar('dilate kernel', 'binary', 2, 10, binary_change)
cv2.createTrackbar('erode iter', 'binary', 1, 10, binary_change)
cv2.createTrackbar('dilate iter', 'binary', 1, 10, binary_change)

all_images = glob.glob("images/*.png")
# filterByArea, minArea, maxArea, filterbyCircularity, minCircularity, filterByConvexity, minConvexity, filterByInertia, minInertiaRatio, maxInertiaRatio
cv2.createTrackbar('image index', 'colored', 0, len(all_images) - 1, image_change)
cv2.createTrackbar('filter by area (T/F)', 'colored', 1, 1, blob_change)
cv2.createTrackbar('min area (10x)', 'colored', 20, 100, blob_change)
cv2.createTrackbar('max area (100x)', 'colored', 50, 100, blob_change)
cv2.createTrackbar('filter by circularity (T/F)', 'colored', 1, 1, blob_change)
cv2.createTrackbar('min circularity (x/100)', 'colored', 70, 100, blob_change)
cv2.createTrackbar('filter by convexity (T/F)', 'colored', 1, 1, blob_change)
cv2.createTrackbar('min convexity (x/100)', 'colored', 70, 100, blob_change)
cv2.createTrackbar('filter by inertia (T/F)', 'colored', 1, 1, blob_change)
cv2.createTrackbar('min inertia (x/100)', 'colored', 10, 100, blob_change)
cv2.createTrackbar('max inertia (x/100)', 'colored', 100, 100, blob_change)

cv2.createTrackbar('primary thresh', 'judged', 200, 255, judge_change)
cv2.createTrackbar('secondary thresh', 'judged', 100, 255, judge_change)
cv2.createTrackbar('pixel thresh', 'judged', 60, 200, judge_change)
cv2.createTrackbar('pixel_ratio', 'judged', 2, 20, judge_change)

image_changed = True
binary_changed = True
blob_changed = True
judge_changed = True

width = 1280
height = 720
crop = True
if crop:
    x_ = 200
    y_ = 100
    w_ = 830
    h_ = 380
    width = int(w_ / 2)
    height = int(h_ / 2)

while True:
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    if image_changed:
        image_index = cv2.getTrackbarPos('image index', 'colored')
        original_image = load_image(all_images[image_index])
        if crop:
            original_image = original_image[y_:y_ + h_, x_:x_ + w_]
        output_original = cv2.resize(original_image, (width, height))

    if image_changed or binary_changed:
        blur_kernel = 2 * cv2.getTrackbarPos('blur kernel (2x+1)', 'binary') + 1
        min_gray_thresh = cv2.getTrackbarPos('min gray thresh', 'binary')
        max_gray_thresh = cv2.getTrackbarPos('max gray thresh', 'binary')
        thresh_kernel = 2 * cv2.getTrackbarPos('thresh kernel (2x+1)', 'binary') + 1
        adaptive_thresh = - cv2.getTrackbarPos('adaptive thresh (-x)', 'binary')
        dilate_kernel = 2 * cv2.getTrackbarPos('dilate kernel', 'binary') + 1
        erode_iter = cv2.getTrackbarPos('erode iter', 'binary')
        dilate_iter = cv2.getTrackbarPos('dilate iter', 'binary')

        blur_image = cv2.medianBlur(original_image, blur_kernel)
        print("blur kernel:", blur_kernel, "min gray thresh:", min_gray_thresh, "max gray thresh:", max_gray_thresh,
              "thresh kernel:", thresh_kernel, "adaptive thresh:", adaptive_thresh)
        gray_image = cv2.cvtColor(blur_image, cv2.COLOR_RGB2GRAY)
        # global thresh
        gray_binary = gray_threshold(gray_image, thresh=(min_gray_thresh, max_gray_thresh))
        # local thresh
        threshold = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                          thresh_kernel, adaptive_thresh)
        # combine them
        combined = np.zeros_like(gray_binary)
        combined[((gray_binary == 255) | (threshold == 255))] = 255
        output_combined = cv2.resize(combined, (width, height))
        kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        combined = cv2.erode(combined, kernel, iterations=erode_iter)
        combined = cv2.dilate(combined, kernel, iterations=dilate_iter)

        output_global_thresh = cv2.resize(gray_binary, (width, height))
        output_local_thresh = cv2.resize(threshold, (width, height))
        output_gray = cv2.resize(combined, (width, height))
        output_binary = np.concatenate((np.concatenate((output_global_thresh, output_local_thresh), axis=0),
                                        np.concatenate((output_combined, output_gray), axis=0)), axis=1)

    if image_changed or blob_changed:
        filterByArea = 1 == cv2.getTrackbarPos('filter by area (T/F)', 'colored')
        minArea = 10 * cv2.getTrackbarPos('min area (10x)', 'colored')
        maxArea = 100 * cv2.getTrackbarPos('max area (100x)', 'colored')
        filterByCircularity = 1 == cv2.getTrackbarPos('filter by circularity (T/F)', 'colored')
        minCircularity = cv2.getTrackbarPos('min circularity (x/100)', 'colored') / 100
        filterByConvexity = 1 == cv2.getTrackbarPos('filter by convexity (T/F)', 'colored')
        minConvexity = cv2.getTrackbarPos('min convexity (x/100)', 'colored') / 100
        filterByInertia = 1 == cv2.getTrackbarPos('filter by inertia (T/F)', 'colored')
        minInertia = cv2.getTrackbarPos('min inertia (x/100)', 'colored') / 100
        maxInertia = cv2.getTrackbarPos('max inertia (x/100)', 'colored') / 100

        print("filter by area:", filterByArea, "min area:", minArea, "max area:", maxArea, "filter by circularity:",
              filterByCircularity, "min circularity:", minCircularity, "filter by convexity", filterByConvexity,
              "min convexity", minConvexity, "filter by inertia:", filterByInertia, "min inertia:", minInertia,
              "max inertia:", maxInertia)
        blob_params = configure_params(filterByArea, minArea, maxArea, filterByCircularity, minCircularity,
                                       filterByConvexity, minConvexity, filterByInertia, minInertia, maxInertia)
        bolb_detector = cv2.SimpleBlobDetector_create(blob_params)

    if judge_changed:
        primary_thresh = cv2.getTrackbarPos('primary thresh', 'judged')
        secondary_thresh = cv2.getTrackbarPos('secondary thresh', 'judged')
        pixel_thresh = cv2.getTrackbarPos('pixel thresh', 'judged')
        pixel_ratio = cv2.getTrackbarPos('pixel_ratio', 'judged')

    if image_changed or binary_changed or blob_changed or judge_changed:
        armour_list = []
        im_with_keypoints = original_image.copy()
        judged_keypoints = original_image.copy()
        # blob detector for circle
        keypoints = bolb_detector.detect(combined)
        # draw the detected circles
        for keypoint in keypoints:
            x = int(keypoint.pt[0])
            y = int(keypoint.pt[1])
            half_width = int(keypoint.size + 10)
            half_height = (keypoint.size + 10) // 2
            y1 = bound_masked_y(y - half_height)
            y2 = bound_masked_y(y + half_height)
            x1 = bound_masked_x(x - half_width)
            x2 = bound_masked_x(x + half_width)
            crop_image = original_image[y1:y2, x1:x2]
            # color judge
            # judge_result = color_judge(crop_image, 30, 100, 60, 2)
            judge_result = color_judge(crop_image, primary_thresh, secondary_thresh, pixel_thresh, pixel_ratio)
            armour_list.append(Armour(x, y, keypoint.size, judge_result))
        for armour in armour_list:
            im_with_keypoints = draw_circle(im_with_keypoints, armour.x, armour.y, armour.size, "None")
            judged_keypoints = draw_circle(judged_keypoints, armour.x, armour.y, armour.size, armour.color)

        output_keypoints = cv2.resize(im_with_keypoints, (width, height))
        output_judged = cv2.resize(judged_keypoints, (width, height))
        output_colored = np.concatenate((output_original, output_keypoints), axis=0)

    if image_changed:
        image_changed = False
    if binary_changed:
        binary_changed = False
    if blob_changed:
        blob_changed = False
    if judge_changed:
        judge_changed = False

    cv2.imshow('colored', output_colored)
    cv2.imshow('binary', output_binary)
    cv2.imshow('judged', output_judged)

cv2.destroyAllWindows()
