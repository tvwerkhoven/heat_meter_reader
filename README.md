# Heat meter through digital seven segment LCD

If you have a (heat) meter with a seven segment display, this guide explains how to add this to domoticz.

## Hardware

- Landis+Gyr Ultraheat
- Camera
- Optional: USB light (for )

### Units

Meter units is in GJ, with 3 decimal digits (e.g. 271.199 GJ). Value we recognize is in MJ because we ignore the decimal point in OCR. Domoticz units are in m^3, with 2 decimal digits (e.g. 3581.92 m^3) -- internally might be stored with higher precision, again not visible through JSON api because of poor interface.

Conversion unit is 13.20772 m^3 gas/GJ, resolution is 0.01 m^3 for total count or 0.001 m^3 for daily count. For electric energy, conversion would be 277.7777 kWh/GJ.

## Software

### Test images / point camera

    raspistill --datetime -v -o cam_%d.jpg -t 0

### Calibrate software

use get_digits.py script: 

    ./get_digits.py --calibrate

or if you want to calibrate on an image taken previously

    ./get_digits.py --calibrate <file>

### Set up domoticz

0. Set gas meter divisor to 1000 (not 100 as is standard)
1. Create dummy hardware
2. On this dummy hardware, create gas meter (Counter Incremental)
   1. Set meter to gas type: utility -> edit -> Type: Gas
   2. Set meter offset: utility -> edit -> Meter offset: <fill in>
3. On this dummy hardware, create power meter (Electric (Instant+Counter)) -- N.B. Currently not supported because of JSON limitations
   1. Set power meter to 'from device': utility -> edit -> "From device", otherwise the consumption is computed from the power
4. Note down idx of two meters created
5. Test updating meters manually using e.g. curl:
   1. get data: `curl -k "https://127.0.0.1:10443/json.htm?type=devices&rid=<idx>"`
   2. update power meter: `curl -k "https://127.0.0.1:10443/json.htm?type=command&param=udevice&idx=<idx>&svalue=<power in Watt>;<usage in Wh>"`
   3. update power meter: `curl -k "https://127.0.0.1:10443/json.htm?type=command&param=udevice&idx=<idx>&svalue=<power in Watt>;<usage in Wh>"`
   4. Once done with testing and you understand how domoticz meters work, delete and recreate meters to start off fresh
6. Update get_digits script
   1. Update meter idx
   2. Update meter unit / type (e.g. gas / power), update conversion factors where necessary

### Run script

Add script to crontab -e, run e.g. every minute

## Alternatives

- https://github.com/jiweibo/SSOCR looks promising
- https://www.unix-ag.uni-kl.de/~auerswal/ssocr/ is very sensitive to noise, didn't work for me
- tesseract / character recognition is overkill given the simplicity of seven segment LCD
- openCV's adaptive thresholding didn't work for me

# References

- https://www.pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python
- https://www.unix-ag.uni-kl.de/~auerswal/ssocr/
- https://github.com/jiweibo/SSOCR
- https://www.swarthmore.edu/NatSci/mzucker1/e27_s2016/install_opencv.html
- https://github.com/arturaugusto/display_ocr
- https://hackaday.com/2013/01/03/ocr-automatically-reads-a-power-meter/