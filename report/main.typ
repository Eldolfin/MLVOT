#set document(
  title: "MLVOT Project Report",
  author: "Oscar",
  date: datetime.today(),
)

#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2cm),
  numbering: "1",
)

#set text(
  font: "New Computer Modern",
  size: 11pt,
  lang: "en"
)

#set heading(numbering: "1.1")

#align(center)[
  #text(size: 20pt, weight: "bold")[MLVOT Project Report]
  
  #v(1cm)
  #text(size: 14pt)[Machine Learning for Visual Object Tracking]
  
  #v(2cm)
  #text(size: 12pt)[Oscar]
  
  #v(0.5cm)
  #text(size: 12pt)[SCIA - 2025]
]

#pagebreak()

#outline(indent: auto)

#pagebreak()

= Introduction
This report details the implementation of a Multi-Object Tracking (MOT) system, evolving from a simple Single Object Tracker (SOT) to a complex Appearance-Aware IoU-Kalman Tracker.

= TP 1: Single Object Tracking with Kalman Filter
== Objective
Implement a Kalman Filter to track a single moving object (ball) detected by a simple color/shape detector.

== Implementation Details
- *State Space:* The state is defined as $(x, y, v_x, v_y)$.
- *Process:* Constant Velocity Model.
- *Results:* The Kalman filter successfully smooths the trajectory and predicts the position when detection is noisy.

= TP 2: IoU-based Tracking (Bounding-Box Tracker)
== Objective
Develop a basic MOT system using Intersection over Union (IoU) and the Hungarian Algorithm for data association.

== Implementation Details
- *Association:* Used `scipy.optimize.linear_sum_assignment` to maximize IoU between existing tracks and new detections.
- *Track Management:* Tracks are created for unmatched detections and deleted after missing for 5 frames.
- *Challenges:* Rapidly moving objects or low frame rate can cause IoU to drop to zero, leading to ID switches.

= TP 3: Kalman-Guided IoU Tracking
== Objective
Improve the IoU tracker by predicting the future bounding box location using a Kalman Filter before association.

== Implementation Details
- *Prediction:* Each track has its own Kalman Filter. Before association, the filter predicts the new centroid.
- *Association:* IoU is calculated between the *predicted* bounding box and the *detected* bounding box.
- *Improvements:* This handles occlusion and faster motion better than pure IoU, as the search window follows the object's velocity.

= TP 4: Appearance-Aware IoU-Kalman Tracker
== Objective
Integrate visual features (ReID) to resolve identity switches and handle long-term occlusions.

== Implementation Details
- *ReID Model:* Used `OSNet` (lightweight) to extract feature vectors from image crops.
- *Cost Function:* A weighted sum of geometric distance (IoU) and visual similarity (Cosine distance).
  $ C_(i j) = alpha dot (1 - "IoU"_(i j)) + beta dot "CosineDist"_(i j) $
- *Result:* Drastically reduced ID switches when objects cross paths.

= Conclusion
The progression from a simple Kalman filter to a ReID-enhanced tracker demonstrates the importance of combining geometric motion models with visual appearance features for robust multi-object tracking.

#bibliography("refs.bib")
