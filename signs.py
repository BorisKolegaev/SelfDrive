import cv2
import numpy as np

cam = cv2.VideoCapture(0)

# ------------------------- подготавливаем исходные картинки -------------------------------------------
right_sign = cv2.imread('resources/right.png')
right_sign = cv2.resize(right_sign, (40, 40))
right_sign = cv2.inRange(right_sign, lowerb=(90, 90, 150), upperb=(255, 255, 255))

man_sign = cv2.imread('resources/man.png')
man_sign = cv2.resize(man_sign, (40, 40))
man_sign = cv2.inRange(man_sign, lowerb=(90, 90, 150), upperb=(255, 255, 255))

left_sign = cv2.imread('resources/left.png')
left_sign = cv2.resize(left_sign, (40, 40))
left_sign = cv2.inRange(left_sign, lowerb=(90, 90, 150), upperb=(255, 255, 255))

up_sign = cv2.imread('resources/up.png')
up_sign = cv2.resize(up_sign, (40, 40))
up_sign = cv2.inRange(up_sign, lowerb=(90, 90, 150), upperb=(255, 255, 255))

tri_sign = cv2.imread('resources/tri.png')
tri_sign = cv2.resize(tri_sign, (40, 40))
tri_sign = cv2.inRange(tri_sign, lowerb=(90, 90, 150), upperb=(255, 255, 255))

noway_sign = cv2.imread('resources/noway.png')
noway_sign = cv2.resize(noway_sign, (40, 40))
noway_sign = cv2.inRange(noway_sign, lowerb=(90, 90, 150), upperb=(255, 255, 255))

stop_sign = cv2.imread('resources/stop.png')
stop_sign = cv2.resize(stop_sign, (40, 40))
stop_sign = cv2.inRange(stop_sign, lowerb=(90, 90, 150), upperb=(255, 255, 255))
# -------------------------------------------------------------------------------------------------------

max_contours = 7


while True:
    ret, frame = cam.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.blur(hsv, (5, 5))

    lower_red_1 = np.array([0, 70, 110])
    upper_red_1 = np.array([10, 255, 255])
    mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    lower_red_2 = np.array([170, 70, 110])
    upper_red_2 = np.array([180, 255, 255])
    mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    maskR = cv2.bitwise_or(mask_1, mask_2)

    lower_blue = np.array([100, 150, 100])
    upper_blue = np.array([140, 255, 255])
    maskB = cv2.inRange(hsv, lower_blue, upper_blue)

    mask = cv2.bitwise_or(maskR, maskB)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=4)

    contours1, hierarchy1 = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours1:
        image_copy1 = frame.copy()
        cnts_sorted = sorted(contours1, key=cv2.contourArea, reverse=True)

        if len(cnts_sorted) > max_contours:
            cnts_sorted = cnts_sorted[:12]
        for c in cnts_sorted:
            x, y, w, h = cv2.boundingRect(c)

            sign_from_image = frame[y:y + h, x:x + w]
            sign_from_image = cv2.resize(sign_from_image, (40, 40))

            sign_from_image = cv2.inRange(sign_from_image, (90, 90, 150), (255, 255, 255))

            cv2.imshow("sign", sign_from_image)
            cv2.imshow("man", man_sign)
            cv2.imshow("right", right_sign)
            cv2.imshow("left", left_sign)
            cv2.imshow("up", up_sign)
            cv2.imshow("tri", tri_sign)
            cv2.imshow("stop", stop_sign)
            cv2.imshow("noway", noway_sign)
            cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (0, 255, 0), 2)
            cv2.imshow("frame", frame)

            counter_right = 0
            counter_man = 0
            counter_left = 0
            counter_up = 0
            counter_tri = 0
            counter_stop = 0
            counter_noway = 0

            for i in range(40):
                for j in range(40):

                    if right_sign[i][j] == sign_from_image[i][j]:
                        counter_right += 1
                    if man_sign[i][j] == sign_from_image[i][j]:
                        counter_man += 1
                    if left_sign[i][j] == sign_from_image[i][j]:
                        counter_left += 1
                    if up_sign[i][j] == sign_from_image[i][j]:
                        counter_up += 1
                    if tri_sign[i][j] == sign_from_image[i][j]:
                        counter_tri += 1
                    if stop_sign[i][j] == sign_from_image[i][j]:
                        counter_stop += 1
                    if noway_sign[i][j] == sign_from_image[i][j]:
                        counter_noway += 1

            if counter_right > 1200:
                print("Знак \"Поворот направо\"")
            if counter_man > 1200:
                print("Знак \"Пешеходный переход\"")
            if counter_left > 1200:
                print("Знак \"Поворот налево\"")
            if counter_up > 1200:
                print("Знак \"Путь вперед\"")
            if counter_stop > 1200:
                print("Знак \"Знак стоп\"")
            if counter_noway > 1200:
                print("Знак \"знак нет пути\"")
            if counter_tri > 1200:
                print("Знак \"треугольник\"")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()