""" Import the necessary functions from vision_utils, window_utils, and input_utils """
from utils.vision_utils import screenshot
from utils.vision_utils import sift_interesting_areas
from utils.vision_utils import orb_interesting_areas
from utils.vision_utils import edge_detection
from utils.vision_utils import histogram_equalization
from utils.vision_utils import crop_image
from utils.vision_utils import template_matching
from utils.vision_utils import mse
from utils.vision_utils import get_image_difference
from utils.vision_utils import SAD

from utils.window_utils import get_window_handle
from utils.window_utils import get_window_position
from utils.window_utils import get_window_size
from utils.window_utils import set_window_position
from utils.window_utils import get_window_size
from utils.window_utils import set_window_size
from utils.window_utils import focus_window

from utils.input_utils import wind_mouse  # function to move mouse like a human
from utils.input_utils import set_mouse_position
from utils.input_utils import click
from utils.input_utils import press_key


import cv2
import time
import numpy as np


""" Add the name of the window you want to capture for cv here """
hwnd_name = "Calculator"  # name of the window to capture
hwnd = get_window_handle(hwnd_name)

# Once hwnd is found, you can use various functions from window_utils to manipulate the window

""" Focus the window and set its position and size for reproducibility """
focus_window(hwnd)
set_window_position(hwnd)


print("Window focused.")
print("Window position:", get_window_position(hwnd))
print("Window size:", get_window_size(hwnd))


""" Main Loop """
fps = 30
while True:
    img = screenshot(hwnd)

    """ Start Processing """
    # manipulate img here using vision_utils functions
    # make some kind of decision based on the processed image and execute it using input_utils functions
    """ End Processing """
    time.sleep(1 / fps)
