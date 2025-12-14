import cv2 as cv
import Detector
from KalmanFilter import KalmanFilter


def main():
    cap = cv.VideoCapture("./video/randomball.avi")

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_fourcc(*"XVID")  # type: ignore
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
            # Predict step
            kalman_filter.predict()
            # Get predicted position (before update)
            predicted_pos = kalman_filter.state[:2].copy()

            # Update step (correct with measurement)
            estimated_pos = kalman_filter.update(center)[:2]

            # Extract scalar values from center array
            center_x = int(center[0, 0])
            center_y = int(center[1, 0])

            # Draw blue rectangle for predicted position
            cv.rectangle(
                frame,
                (int(predicted_pos[0]) - 15, int(predicted_pos[1]) - 15),
                (int(predicted_pos[0]) + 15, int(predicted_pos[1]) + 15),
                (255, 0, 0),  # Blue in BGR
                2,
            )
            # Draw red rectangle for estimated position
            cv.rectangle(
                frame,
                (int(estimated_pos[0]) - 15, int(estimated_pos[1]) - 15),
                (int(estimated_pos[0]) + 15, int(estimated_pos[1]) + 15),
                (0, 0, 255),  # Red in BGR
                2,
            )
            # Draw green circle for detected position
            cv.circle(frame, (center_x, center_y), 10, (0, 255, 0), 2)
            trajectory.append((center_x, center_y))

        for i in range(1, len(trajectory)):
            cv.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()


if __name__ == "__main__":
    main()
