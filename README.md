# OpenCV Features

![Picture](https://github.com/GorokhovSemyon/Simple_bash_scripst-MacOS-/blob/develop/materials/bin_bash.png)

## Introduction

OpenCV ([Open Source Computer Vision Library](https://opencv.org/)) is an open library for working with computer vision algorithms, machine learning and image processing. It is written in C++, but it also exists for Python, JavaScript, Ruby and other programming languages. Works on Windows, Linux and macOS, iOS and Android.

## Instructions for use

First of all, to use all the programs you will need installed on your computer:
- [`mediapipe`](https://pypi.org/project/mediapipe/) - to improve recognition and processing capabilities
- [`OpenCV`](https://opencv.org/) - for processing incoming video/image
- [`python`](https://www.python.org/) - for the general assembly of the project
- [`numpy`](https://numpy.org/) - for mathematical calculations
- [`time`](https://docs.python.org/3/library/time.html ) - to get the system time and FPS

### Photo processing

Some image processing methods (drawing geometric shapes, adding text to the image, blurring, redefinition, color conversion) are presented in the file `src/photo_processing.py `

### Outline selection

For the task of selecting contours in the incoming image, a file can be used `src/finding_contors.py `

### Color formats

The color formats can be changed by means of OpenCV, as an example, a file has been created `src/colors_formats.py `

### Bitwise operations and masks

In addition to the above, OpenCV allows you to process specified areas selected from the entire image, which can be extremely useful, an example of this is presented in the file `src/bitwise_operations_and_masks.py `

### Video processing

In addition to processing static frames, it is possible to process video images, both preset and received in real time from the camera, an example is a file `src/video_processing.py `

### Selection of objects (faces) in the frame

A more useful application of all of the above can be carried out in the form of the task of determining people's faces in real time, which is implemented in the file `src/finding_faces.py `

## Mono-measurement of the distance to objects (faces) using a neural network

The processing speed in this task largely depends on the hardware "stuffing" of the computer. As a result, the program measures the distance to objects (persons) in real time and displays a distance map of the observed scene. The result is a view stream with information, the implementation is in the file `src/main.py `