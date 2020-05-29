#     The Starburst Project - Hardik Shrestha - hardik.ho@gmail.com
#     Computer-Vision (CV) Project
#     Created on May 29, 2020
#     github.com/Hardik-S | hardik-s.github.io

import cv2 as cv
import numpy as np
import time as t

# global vars #
x = 0
y = 0
counter = 0

# boolean
drawing = True
endAll = False
mouseClick = False

# start time
t0 = t.time()

# initiates capture, turns on camera
capture = cv.VideoCapture(0)


def take_input(event, x1, y1, flag, param):
    global x, y, mouseClick
    if event == cv.EVENT_LBUTTONDOWN:
        x = x1
        y = y1
        mouseClick = True

        # burner args
        flag = None
        param = None


cv.namedWindow("Canvas")
cv.setMouseCallback("Canvas", take_input, param=None)

# initial loop #

while True:

    # captures camera footage, flips horizontally, converts to grayscale
    blank, input_image = capture.read()
    input_image = cv.flip(input_image, 1)
    grayscale_input_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)

    # crop input
    cropped_input_image = input_image[70:400, 0:626]

    # show user
    cv.imshow("Canvas", cropped_input_image)

    # resize and move window to align with 'Mask' window later
    cv.resizeWindow("Canvas", 626, 330)
    cv.moveWindow("Canvas", 0, 125)

    # on esc or left click, breaks loop and starts main program
    if mouseClick or cv.waitKey(30) == 27:
        cv.destroyAllWindows()
        break

# optical flow start #

# creates array using float values of object co-ordinates
old_points = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)

# creates mask
mask = np.zeros_like(input_image)

while True:

    # captures camera footage, flips horizontally, converts to grayscale
    blank, new_input_image = capture.read()
    new_input_image = cv.flip(new_input_image, 1)
    new_grayscale = cv.cvtColor(new_input_image, cv.COLOR_BGR2GRAY)

    # uses open cv to draw at base (highest) resolution [adjust using maxLevel, higher = blurrier]
    # can adjust criteria based on CPU - [100, 2 produces visually decent mask though]
    new_points, status, err = cv.calcOpticalFlowPyrLK(grayscale_input_image, new_grayscale, old_points, None,
                                                      maxLevel=0,
                                                      criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 100, 2))

    # extracts co-ordinate values and turns into integers
    # unfortunately cannot be done in one step
    for i, j in zip(old_points, new_points):
        x, y = j.ravel()
        a, b = i.ravel()
        x = int(x)
        y = int(y)
        a = int(a)
        b = int(b)

        # flips between drawing and not drawing on 'S' press
        if cv.waitKey(1) == ord('s'):
            if drawing:
                drawing = False
            elif not drawing:
                drawing = True

        # zeroes all values, clearing the board
        elif cv.waitKey(1) == ord('c'):
            mask = np.zeros_like(new_input_image)

        # traces line based on object movement
        if drawing:
            mask = (cv.line(mask, (a, b), (x, y), (247, 255, 130), 3))

        # img, center, radius, BGR colour [, thickness[, lineType[, shift]]]
        cv.circle(new_input_image, (x, y), 3, (247, 255, 130), -1)

    # define new image with the raw drawing masked on top at 50% opacity
    new_input_image = cv.addWeighted(mask, 0.5, new_input_image, 0.7, 0)

    # counts iterations through loop, calculates Frames Per Second
    counter = counter + 1
    FPS = "FPS: "+str(round((counter/(t.time()-t0)), 1))

    # displays instructions
    cv.putText(mask, "FPS: 19.2", (10, 100),
               cv.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255))
    cv.putText(mask, "'S' to start/stop", (10, 125),
               cv.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255))
    cv.putText(mask, "'C' to clear", (10, 150),
               cv.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255))
    cv.putText(mask, "'Esc' to exit", (10, 175),
               cv.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255))

    # crops raw and mask windows
    cropped_mask = mask[70:400, 0:626]
    cropped_raw = new_input_image[70:400, 0:626]

    # show two windows with raw and mask
    cv.imshow("Mask", cropped_mask)
    cv.imshow("Raw", cropped_raw)

    # resizes and moves raw and mask windows
    cv.resizeWindow("Mask", 626, 330)
    cv.moveWindow("Mask", 650, 125)
    cv.resizeWindow("Raw", 626, 330)
    cv.moveWindow("Raw", 0, 125)

    # creates new points by essentially duplicating the old points
    grayscale_input_image = new_grayscale.copy()
    old_points = new_points.reshape(-1, 1, 2)

    # checks every ~1s (based on CPU) to print status and error
    # if error is too high, calls for end
    if counter % 30 == 0:
        print('---', counter / 30, '---')
        print("time:", t.strftime("%H:%M:%S", t.localtime()))
        print(FPS)
        print("status: ")
        if status == 1:
            print("paintbrush detected")
        if status == 0:
            print("paintbrush cannot be detected")
        if err >= .8:
            print("High error, slow down!")
        if err > 1.6:
            endAll = True

    # esc key pressed = break
    if cv.waitKey(1) & 0xff == 27:
        break

    # if error too high, break.
    # prints time passed from launch to crash
    if endAll:
        print("Error was too large.")
        print("Lasted ", round((t.time()-t0), 1), " seconds total.")
        break

# closes both windows and turns off camera
cv.destroyAllWindows()
capture.release()
