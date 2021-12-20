import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    copy_frame = np.copy(frame)

    frame = cv2.GaussianBlur(frame, (7, 7), 0)

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    cnts_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

    cv2.drawContours(image=copy_frame, contours=cnts_sorted[:1], contourIdx=-1, color=(0, 255, 0), thickness=2,
                     lineType=cv2.LINE_AA)

    cv2.imshow("frame", copy_frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()




