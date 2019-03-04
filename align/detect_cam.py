import cv2

from detector import Face_detect
from align.visualization_utils import show_results
import numpy as np
if __name__ == '__main__':
    from PIL import Image
    import time

    face_detect=Face_detect()
    img = Image.open('d:/guo.jpg')  # modify the image path to yours
    #cpu需要570ms
    #gpu 需要256ms

    capture = cv2.VideoCapture(0)
    while True:
        ret, frame = capture.read()

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        start = time.time()
        bounding_boxes, landmarks = face_detect.detect_faces(
            image)  # detect bboxes and landmarks for all faces in the image
        print('detect time', time.time() - start)
        image = show_results(image, bounding_boxes, landmarks)  # visualize the results
        img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        cv2.imshow("OpenCV", img)
        if cv2.waitKey(1) == ord('q'):
            break
            self.capture.release()
    # for i in range(10):
    #     start=time.time()
    #     bounding_boxes, landmarks = face_detect.detect_faces(img)  # detect bboxes and landmarks for all faces in the image
    #     print('detect time',time.time()-start)
    # img=show_results(img, bounding_boxes, landmarks)  # visualize the results
    # img.show()