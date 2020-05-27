import cv2 as cv
import numpy as np

# global
x, y, h = 0, 0, -1

# boolean
drawing = True

# initiates capture, turns on camera
capture = cv.VideoCapture(0)


def take_input(event, x1, y1, flag, param):
    global x, y, h
    if event == cv.EVENT_LBUTTONDOWN:
        x = x1
        y = y1
        h = 1
        flag = None
        param = None


cv.namedWindow("canvas")
cv.setMouseCallback("canvas", take_input, param=None)

# main loop #

# captures camera footage, flips horizontally, converts to grayscale
while True:

    blank, input_image = capture.read()
    input_image = cv.flip(input_image, 1)
    grayscale_input_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)

    # show user
    cv.imshow("canvas", input_image)

    # on esc, ends program
    if h == 1 or cv.waitKey(30) == 27:
        cv.destroyAllWindows()
        break

# optical flow start #

# creates array using float values of object co-ordinates
old_points = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)

# projects mask onto canvas
mask = np.zeros_like(input_image)

# captures camera footage, flips horizontally, converts to grayscale
while True:
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

        # flips between drawing and not drawing on 'Q' press
        if cv.waitKey(1) & 0xff == ord('q'):
            if drawing:
                drawing = False
            elif not drawing:
                drawing = True

        # zeroes all values, clearing the board
        elif cv.waitKey(1) == ord('n'):
            mask = np.zeros_like(new_input_image)

        # traces line based on object movement
        if drawing:
            mask = (cv.line(mask, (a, b), (x, y), (247, 255, 130), 6))

        # img, center, radius, BGR colour [, thickness[, lineType[, shift]]]
        cv.circle(new_input_image, (x, y), 6, (247, 255, 130), -1)

    # define new image with the raw drawing masked on top at 50% opacity
    new_input_image = cv.addWeighted(mask, 0.5, new_input_image, 0.7, 0)

    # instructions
    cv.putText(mask, "'Q' to start/drawing | 'N' to clear | 'esc' to exit", (10, 50),
               cv.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255))

    # show two windows with raw and mask
    cv.imshow("Raw", new_input_image)
    cv.imshow("Mask", mask)

    # creates new points by essentially duplicating the old points
    grayscale_input_image = new_grayscale.copy()
    old_points = new_points.reshape(-1, 1, 2)

    # esc key = break
    if cv.waitKey(1) & 0xff == 27:
        break

# on esc, breaks and closes both windows and turns off camera
cv.destroyAllWindows()
capture.release()
