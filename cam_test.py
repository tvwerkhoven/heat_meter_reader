#!/usr/bin/env python3

from picamera import PiCamera
from time import sleep
import time
from fractions import Fraction

### Trying - 

camera = PiCamera()#sensor_mode=4, resolution=(1296,972), framerate=Fraction(1))

print("sleeping..")
sleep(2)
#camera.awb_mode='off'
#camera.exposure_mode = 'off'
print("capture 1")
camera.capture('dark_ip_{}.jpg'.format(time.time()), use_video_port=False)
print("capture 2")
camera.capture('dark_vp_{}.jpg'.format(time.time()), use_video_port=True)
print("capture done")

### Trying - this sets the framerate to 1 and exposure time as well. Binning not visible

# camera = PiCamera(sensor_mode=4, framerate=Fraction(1, 6))
# camera.shutter_speed = 6000000
# camera.iso = 800

# sleep(30)
# camera.awb_mode='off'
# camera.exposure_mode = 'off'
# # Finally, capture an image with a 6s exposure. Due
# # to mode switching on the still port, this will take
# # longer than 6 seconds
# camera.capture('dark_{}.jpg'.format(time.time()))



### This works
# dark3.jpg - dark_1527191565.633295.jpg

# camera = PiCamera(framerate=Fraction(1, 6))
# camera.shutter_speed = 6000000
# camera.iso = 800

# sleep(30)
# camera.awb_mode='off'
# camera.exposure_mode = 'off'
# # Finally, capture an image with a 6s exposure. Due
# # to mode switching on the still port, this will take
# # longer than 6 seconds
# camera.capture('dark_{}.jpg'.format(time.time()))


