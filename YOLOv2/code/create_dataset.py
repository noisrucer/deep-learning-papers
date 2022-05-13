import cv2
import time
import os.path as osp

data_dir = './data/images'
labels = {
    0: 'a',
    1: 'b',
    2: 'c',
    3: 'd',
    4: 'e',
    8: 'i',
}

cap = cv2.VideoCapture(0) # webcam
cap.set(3, 416) # Width: id 3 as 640
cap.set(4, 416) # Height: id 4 as 480
cap.set(10, 300) # brightness: id 10 as 100

frame_size = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

f_width, f_height = frame_size
print("Frame width: {}".format(frame_size[0]))
print("Frame height: {}".format(frame_size[1]))

start = time.time()

num_samples_per_label = 41
capture_cnt = 31
cur_label = 4

while True:
    success, frame = cap.read()

    cv2.putText(frame, "label: " + str(labels[4]), (f_width-120, f_height-60),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=(0, 255, 0), thickness=2)

    cv2.putText(frame, "count: " + str(int(time.time()-start)), (f_width-130, f_height-20),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=(0, 0, 255), thickness=2)


    if time.time() > start + 4:
        _, frame_cap = cap.read()
        cv2.imwrite(osp.join(data_dir, 'label{}_{}.png'.format(str(cur_label), str(capture_cnt))), frame_cap)
        start = time.time()
        capture_cnt += 1

    if time.time() - start < 0.5:
        _, frame_cap = cap.read()
        cv2.putText(frame_cap, "Captured!", (f_width-150, f_height-20),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.7, color=(0, 255, 255), thickness=2)

        cv2.putText(frame_cap, "Total count: {}".format(capture_cnt-1), (f_width-200, f_height-60),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.7, color=(0, 255, 255), thickness=2)

        cv2.imshow('webcam', frame_cap)
    else:
        cv2.imshow("webcam", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


