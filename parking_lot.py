import car_detector
import cv2
import imutils

inference = car_detector.CarDetector("detect.tflite", "coco_labels.txt")
cap = cv2.VideoCapture('video.mp4')
count = 0

while True:
    ret, img = cap.read()
    # if video stopped playing, quit
    if ret == False:
        break

    count = count + 1
    if count % 5 != 0:
        continue

    areas = [
        # y1,  y2,  x1,  x2
        [608, 700, 397, 565],
        [451, 478, 538, 605],
        [405, 419, 302, 380],
        [470, 530, 345, 432]
    ]

    for area in areas:
        crop = img[area[0]:area[1], area[2]:area[3]]
        # cv2.imshow('crop', crop)
        ret = inference.detect2(crop)
        # print(round(ret[0], 2))
        if ret >= 0.3:  # FREE : BLUE
            img = cv2.rectangle(img, (area[2], area[0]), (area[3], area[1]), (255, 0, 0), 3)
        else:  # BUSY : RED
            img = cv2.rectangle(img, (area[2], area[0]), (area[3], area[1]), (0, 0, 255), 5)

    cv2.imshow('image', img)
    cv2.waitKey(1)
