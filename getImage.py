import cv2

path = "C:/Users/47925/Desktop/Data_sign_language/Valid/Z/"
times = 200 # 训练集一千张，验证集两百

capture = cv2.VideoCapture(0)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
top_left_x = (width - 224) // 2
top_left_y = (height - 224) // 2
index = 1
for i in range(times):
    print(i)
    ret, frame = capture.read()
    if ret is True:
        cropped = frame[top_left_y:top_left_y + 224, top_left_x:top_left_x + 224]
        cv2.imshow("frame", cropped)
        cv2.imwrite(path + str(i) + ".jpg", cropped)
        c = cv2.waitKey(50)
        if c == 27:
            break
    else:
        break

cv2.destroyAllWindows()
capture.release()
