print("AAAA")
import cv2
import os

print(os.path.realpath())
cap1 = cv2.VideoCapture("C:\\Users\\cksdi\\PycharmProjects\\anogan\\1.MP4")
cap2 = cv2.VideoCapture("C:\\Users\\cksdi\\PycharmProjects\\anogan\\2.MP4")
cap3 = cv2.VideoCapture("C:\\Users\\cksdi\\PycharmProjects\\anogan\\3.MP4")

## 비디오가 정상적으로 열렸는지 확인

i = 0
l1 = cap1.getll(cv2.CAP_PROP_FRAME_COUNT)
l2 = cap2.get(cv2.CAP_PROP_FRAME_COUNT)
l3 = cap3.get(cv2.CAP_PROP_FRAME_COUNT)
a = []
while (cap1.isOpened):
    ret, frame = cap1.read()
    if ret and (i in a):
        cv2.imwrite("./"+str(i)+".png", frame)
    else:
        break
    i += 1
cap1.release()

cv2.destroyAllWindows()
["bottle", "drink", "penut_butter", "light", "black_cup", "salt", "calc", "sponge", "spam", "cleaner", "book", "white_cup", "scissors", "banana", "spoon", "pen", "downy"]