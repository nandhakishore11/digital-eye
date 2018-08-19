# Digital Eye for Visually Impaired
## About
Speaks out to the user what is around them by using Tiny YOLO v2.
## Motivation
I created this project as a mini-project in my college. This aims to help people with little or no sight by telling them out what is around them. My contribution to society.

## Installation

Connect a PiCamera to a Raspberry Pi 3. Install all required dependencies using PIP. Now run
```
python3 final_raspi.py
```
You'll begin to see text on the command window. My team has implemented TTS separately. This output will be piped to that TTS engine and will be spoken out to the user.

## Understanding
In order to understand whats happening here, how the image is analyzed and is found out for objects, you need to learn the following.
```
Deep Neural Networks
Convolutional Neural Networks
YOLO (Only the architecture. Not darknet)
```
## Helpers
I sought help from various sources other than online courses. They really helped me understand what's happening to deduce boxes from YOLO's predictions. They are
* [Christopher Bourez's Explanation](http://christopher5106.github.io/object/detectors/2017/08/10/bounding-box-object-detectors-understanding-yolo.html)  - AWESOME Explanation. You've to check it out!! The best in the internet.
* [YAD2K](https://github.com/allanzelener/YAD2K) 
* [DarkFlow](https://github.com/thtrieu/darkflow)
