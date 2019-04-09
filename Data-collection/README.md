# Image Data Collection

## Description

This project is for Image/Video data collection from a camera which is plugged to a PC.

Please set ```capture_fps```, ```i(pic saving freqency)```, ```rec_time```, ```resolution_scale```before capturing image data.
The function will store video(.avi) and pictures(.jpg) simultaneously.
Video is saved to "Data-video" folder and pictures are saved to "Data-pic" folder.

## Installation

* Python 3 recommended
    Python 2.7 will reach the end of its life on January 1st, 2020.
* Library Dependencies:
    * numpy
    * opencv 3.4.1

## Usage

1. Set up data collection environment.(indoor scenario)
2. Connect a camera to a PC with USB or other interface.
3. Run "data_collection.py" in powershell or cmd. (make sure PC will not auto lock)
    ```python
    python .\data_collection.py
    ```
4. Find the data in file "Data-video" and "Data-pic".

## License

## Reference
