import Detector
from KalmanFilter import KalmanFilter
import cv2 as cv

def main():
    cap = cv.VideoCapture('./video/randomball.avi')
    kalman_filter = KalmanFilter(dt=0.1, u_x=1, u_y=1, std_acc=1, x_dt_meas=0.1, y_dt_meas=0.1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        centers = Detector.detect(frame)
        if centers == []:
            print("No balls detected !")
        pred = kalman_filter.predict(centers[0])
        kalman_filter.update(centers[0])

        

    cap.release()

if __name__ == "__main__":
    main()
