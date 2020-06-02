# The Starburst Project

## Preview
![](https://github.com/Hardik-S/the-starburst-project/blob/master/Media/Drawing%20The%20Star.gif)

## Summary
Uses open-computer-vision (Open-cv) to track the object movement of a starburst. The image is converted to grayscale and flipped to mirror the movement of the object. The co-ordinates are tracked using numpy and used to draw a line following the object path. 

[Relevant Doc](https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html) from [OpenCV](https://docs.opencv.org/2.4/index.html)

## Instructions 

- Download _the-starbust-project.py_ and run it using an environment of your choice. 
- If using External Webcam, ensure that it is set as the default. 
  - Alternatively, change line 26 to `capture = cv.VideoCapture([0/1/2])` dependent on the camera you would like to use.
  - Alternatively, disable the local camera's driver directly forcing the program to find the first working camera.
 - On run, Canvas will open up. Navigate cursor to relevant starburst (pointer) and left click. 
 - Raw and Mask windows will open up. Paint the air with your starburst! 
 
 ### Specific Instructions 
 
 - Long-press 's' key to start/stop drawing
 - Long-press 'c' key to clear the canvas
 - Long-press 'esc' key to close all windows. 
 - If program keeps crashing with Error 77, comment lines 171-172
 - If program uses too many resources, read line 84, adjust maxLevel and criteria as necessary
 - For advanced stat tracking, set stats on line 20 to True
 
 ### Peronsalization
 
 - Basic: edit the BGR values anywhere for your line colour of choice.
 - Advanced: read the comments
 
 ## Future Additions
 - Colour change on key press (b = blue, r = red, x = swap to next) 
 - Automatic screenshot of project on end
 - Automatic recording of project on key press (v = video (tape)) 
 - Port to Web using Django? 
 
 ## Another Example with my Mom
 
 ![](https://github.com/Hardik-S/the-starburst-project/blob/master/Media/mamu%20rose.png)


## Another Example wih My Girlfriend

![](https://github.com/Hardik-S/the-starburst-project/blob/master/Media/ananu%20love.png)
