import cv2

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (0, 97 , 38), (71, 179, 186))

    result = cv2.bitwise_and(img, img, mask = mask)

    cv2.imshow("Original", img)
    cv2.imshow("Result", result)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()