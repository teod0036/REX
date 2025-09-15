# This script shows how to open a camera the picamera2 module and grab frames and show these.
# Kim S. Pedersen, 2023

from picamera2 import Picamera2
import time
picam2 = Picamera2()
capture_config = picam2.create_still_configuration()
picam2.start(show_preview=True)
time.sleep(1)
picam2.switch_mode_and_capture_file(capture_config, "testfocal.jpg")