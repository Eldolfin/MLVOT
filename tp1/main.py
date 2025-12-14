import cv2 as cv
import Detector
from KalmanFilter import KalmanFilter


def main():
    cap = cv.VideoCapture("./video/randomball.avi")

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_fourcc(*"XVID")
    out = cv.VideoWriter("output.avi", fourcc, 60.0, (width, height))

    kalman_filter = KalmanFilter(
        dt=0.1, u_x=1, u_y=1, std_acc=1, x_dt_meas=0.1, y_dt_meas=0.1
    )

    trajectory = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        centers = Detector.detect(frame)
        center = centers[0]

        if centers:
            kalman_filter.predict()
            pred = kalman_filter.update(center)
            # __import__("ipdb").set_trace()
            cv.rectangle(
                frame,
                (int(center[0]) - 15, int(center[1]) - 15),
                (int(center[0]) + 15, int(center[1]) + 15),
                (255, 0, 0),
                2,
            )
            cv.rectangle(
                frame,
                (int(pred[0]) - 15, int(pred[1]) - 15),
                (int(pred[0]) + 15, int(pred[1]) + 15),
                (0, 0, 255),
                2,
            )
            cv.circle(frame, (int(center[0]), int(center[1])), 10, (0, 255, 0), 2)
            trajectory.append((int(center[0]), int(center[1])))

        for i in range(1, len(trajectory)):
            cv.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()


if __name__ == "__main__":
    main()
