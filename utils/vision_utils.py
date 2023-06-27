import win32gui
import pyautogui
import pydirectinput
import cv2
import numpy as np


def screenshot(hwnd):
    """
    Capture a screenshot of the specified window identified by its window handle (hwnd).

    Parameters:
        hwnd (int): The window handle (HWND) of the window to capture.

    Returns:
        numpy.ndarray: The screenshot image as a NumPy array in RGB format.
    """
    # Retrieve the dimensions of the client area
    client_rect = win32gui.GetClientRect(hwnd)

    # Convert client coordinates to screen coordinates
    left, top, right, bottom = client_rect
    left, top = win32gui.ClientToScreen(hwnd, (left, top))
    right, bottom = win32gui.ClientToScreen(hwnd, (right, bottom))

    # Capture the screenshot of the visible client area
    screenshot = pyautogui.screenshot(region=(left, top, right - left, bottom - top))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)


def crop_image(img, x, y, w, h):
    """
    Crop the specified image to the specified dimensions.

    Parameters:
        img (numpy.ndarray): The image to crop.
        x (int): The x-coordinate (horizontal position) of the top-left corner of the crop area in pixels.
        y (int): The y-coordinate (vertical position) of the top-left corner of the crop area in pixels.
        w (int): The width of the crop area in pixels.
        h (int): The height of the crop area in pixels.

    Returns:
        numpy.ndarray: The cropped image as a NumPy array in RGB format.
    """
    # typecast all values to int
    x, y, w, h = int(x), int(y), int(w), int(h)
    return img[y : y + h, x : x + w]


def template_matching(image, template, threshold=0.8):
    """
    Performs template matching on the specified image using the specified template.

    Parameters:
        image (numpy.ndarray): The image to search in.
        template (numpy.ndarray): The template to search for.
        threshold (float): The threshold above which a match is considered valid. Defaults to 0.8.

    Returns:
        tuple: A tuple (x, y) representing the coordinates of the top-left corner of the match in pixels.
    """
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val >= threshold:
        return max_loc
    return None


def get_image_difference(img1, img2):
    if img1.shape == img2.shape:
        # Compute the absolute difference between the two images
        diff_image = cv2.absdiff(img1, img2)

        # Split the difference image into its color channels
        b, g, r = cv2.split(diff_image)

        # Combine the color channels into a single image
        zeros = np.zeros(diff_image.shape[:2], dtype=np.uint8)
        diff_image = cv2.merge((b, zeros, zeros))

        return diff_image

    else:
        # Invalid images
        raise ValueError("The images must be of same shape")


def SAD(img1, img2):
    # sum of absolute differences implementation
    # Ensure both images are the same shape
    assert img1.shape == img2.shape, "Images must be the same shape."

    return np.sum(np.abs(img1 - img2))


def mse(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    h, w = img1.shape
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff**2)
    mse = err / (float(h * w))
    return mse


import cv2
import numpy as np


def sift_interesting_areas(image, threshold=10):
    """
    Perform SIFT (Scale-Invariant Feature Transform) on the image and threshold interesting areas.

    Parameters:
        image (numpy.ndarray): The image to analyze as a NumPy array in RGB format.
        threshold (int): The threshold value to determine interesting areas (default: 10).

    Returns:
        numpy.ndarray: A binary thresholded image with interesting areas in white and non-interesting areas in black.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(gray, None)
    mask = np.zeros_like(gray)
    mask = cv2.drawKeypoints(
        gray, keypoints, mask, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    # _, thresholded = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    return mask


def orb_interesting_areas(image, threshold=1000):
    """
    Perform ORB (Oriented FAST and Rotated BRIEF) feature detection on the image.

    Parameters:
        image (numpy.ndarray): The image to analyze as a NumPy array in RGB format.

    Returns:
        numpy.ndarray: A binary thresholded image with interesting areas in white and non-interesting areas in black.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create()
    keypoints = orb.detect(gray, None)
    mask = np.zeros_like(gray)
    mask = cv2.drawKeypoints(
        gray, keypoints, mask, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    # _, thresholded = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    return mask


def edge_detection(image, threshold1=100, threshold2=200):
    """
    Perform edge detection on the image using the Canny edge detection algorithm.

    Parameters:
        image (numpy.ndarray): The image to analyze as a NumPy array in RGB format.
        threshold1 (int): The first threshold for the hysteresis procedure (default: 100).
        threshold2 (int): The second threshold for the hysteresis procedure (default: 200).

    Returns:
        numpy.ndarray: A binary edge image with detected edges in white and non-edge areas in black.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    return edges


def histogram_equalization(image):
    """
    Perform histogram equalization on a color image.

    Parameters:
        image (numpy.ndarray): The color image to equalize.

    Returns:
        numpy.ndarray: The equalized image.
    """
    # Split the image into individual color channels
    b, g, r = cv2.split(image)

    # Apply histogram equalization to each channel
    b_equalized = cv2.equalizeHist(b)
    g_equalized = cv2.equalizeHist(g)
    r_equalized = cv2.equalizeHist(r)

    # Merge the equalized channels back into a color image
    equalized = cv2.merge([b_equalized, g_equalized, r_equalized])

    return equalized


import cv2


def compare_images(image1, image2, method="mse", threshold=None):
    """
    Compare two images using the specified method and return a boolean indicating similarity.

    Parameters:
        image1 (numpy.ndarray): The first image as a NumPy array.
        image2 (numpy.ndarray): The second image as a NumPy array.
        method (str): The image comparison method to use. Options: 'mse', 'ssim', 'ncc', 'histogram', 'feature'.
        threshold (float): The threshold value for similarity. Images with similarity above this threshold are considered similar.

    Returns:
        bool: True if the images are considered similar, False otherwise.
    """
    if method == "mse":
        # Calculate Mean Squared Error (MSE) including color information
        mse = ((image1 - image2) ** 2).mean()
        if threshold is None:
            return mse == 0
        else:
            return mse <= threshold

    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    if method == "ssim":
        # Calculate Structural Similarity Index (SSIM)
        ssim = cv2.compareSSIM(gray1, gray2)
        if threshold is None:
            return ssim == 1
        else:
            return ssim >= threshold

    elif method == "ncc":
        # Calculate Normalized Cross-Correlation (NCC)
        ncc = cv2.matchTemplate(gray1, gray2, cv2.TM_CCORR_NORMED)
        if threshold is None:
            return ncc.max() == 1
        else:
            return ncc.max() >= threshold

    elif method == "histogram":
        # Calculate histogram intersection distance
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        intersection = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
        if threshold is None:
            return intersection == hist1.sum()
        else:
            return intersection >= threshold

    elif method == "feature":
        # Use feature matching (SIFT) to count the number of matches
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = [m1 for m1, m2 in matches if m1.distance < 0.75 * m2.distance]
        if threshold is None:
            return len(good_matches) > 0
        else:
            return len(good_matches) >= threshold

    else:
        raise ValueError(
            "Invalid method. Available options: 'mse', 'ssim', 'ncc', 'histogram', 'feature'"
        )


def compare_frame_with_template(frame, template, method="ccoeff", threshold=0.8):
    """
    Compare a frame with a template using template matching techniques.

    Parameters:
        frame (numpy.ndarray): The input frame as a NumPy array.
        template (numpy.ndarray): The template image as a NumPy array.
        method (str): The template matching method to use. Options: 'sqdiff', 'sqdiff_normed', 'ccorr', 'ccorr_normed', 'ccoef', 'ccoef_normed'.
        threshold (float): The threshold value for template matching similarity. Images with similarity above this threshold are considered a match.

    Returns:
        tuple: A tuple (x, y) representing the coordinates of the best match near the center of the detected region. Returns None if no match is found.
    """
    # Convert images to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Choose the template matching method
    methods = {
        "sqdiff": cv2.TM_SQDIFF,
        "sqdiff_normed": cv2.TM_SQDIFF_NORMED,
        "ccorr": cv2.TM_CCORR,
        "ccorr_normed": cv2.TM_CCORR_NORMED,
        "ccoef": cv2.TM_CCOEFF,
        "ccoef_normed": cv2.TM_CCOEFF_NORMED,
    }

    method_code = methods.get(method)
    if method_code is None:
        raise ValueError(
            "Invalid method. Available options: 'sqdiff', 'sqdiff_normed', 'ccorr', 'ccorr_normed', 'ccoef', 'ccoef_normed'"
        )

    # Perform template matching
    result = cv2.matchTemplate(gray_frame, gray_template, method_code)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # Check if similarity exceeds the threshold
    if max_val >= threshold:
        # Calculate the coordinates of the match near the center
        h, w = template.shape[:2]
        match_x = max_loc[0] + w // 2
        match_y = max_loc[1] + h // 2
        return match_x, match_y

    return None
