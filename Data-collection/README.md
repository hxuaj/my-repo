# Data Collection

## Description

This project is for Image/Video data collection from a camera which is plugged to a PC or a raspberry.

The function will store video(.avi) and pictures(.jpg) simultaneously to different folders.

Please set ```capture_fps```, ```i(pic saving freqency)```, ```rec_time```, ```resolution_scale```before capturing video and image data.


## Installation

* Python 3 recommended
* Library Dependencies:
    * numpy
    * opencv 3.4.1

## Usage

1. Set up data collection environment.
2. Connect a camera to a PC or a raspberry pi with USB or CSI.
3. Run "data_collection.py" in powershell or cmd.
    ```python
    python .\data_collection.py [ -f <capture_fps>][-i <pic_save_freq>][-r <rec_time>][-rs <resolution_scale>]
    ```
    Run ```python .\data_collection.py -h``` for help.

4. Video is saved to "Data-video" folder and pictures are saved to "Data-pic" folder.

## License

## Reference
