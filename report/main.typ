#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import fletcher.shapes: diamond

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
This project aims to implement and analyze various object tracking algorithms, ranging from a simple Single Object Tracker (SOT) to a sophisticated Appearance-Aware Multi-Object Tracker (MOT). The goal is to understand the trade-offs between computational efficiency, geometric heuristics, and deep learning-based appearance features in maintaining object identities over time.

We progressively build a tracking system, evaluating each iteration against the `ADL-Rundle-6` dataset.

= Dataset and Experimental Setup
== Data Overview
The tracking algorithms are evaluated on the `ADL-Rundle-6` sequence from the MOT15 benchmark. This dataset features a busy pedestrian street captured by a static camera.

- *Detections (`det.txt`):* Provided pre-computed detections including bounding boxes $(x, y, w, h)$ and confidence scores.
- *Ground Truth (`gt.txt`):* Used for quantitative evaluation (though primarily qualitative analysis is presented here).
- *Challenges:* The dataset includes frequent occlusions, crowded scenes with crossing trajectories, and variations in object scale.

#figure(
  rect(width: 100%, height: 6cm, fill: luma(240))[
    #align(center + horizon)[*Placeholder: Sample frames from ADL-Rundle-6*]
  ],
  caption: [Visual characteristics of the dataset: Crowded pedestrian street.]
)

= TP 1: Single Object Tracking with Kalman Filter
== Objective
The objective of this first step is to implement a Kalman Filter to track a single moving object. In this specific experiment, we track a ball in a video sequence (`randomball.avi`) using a color/contour-based detector.

== Methodology
We utilize a discrete linear Kalman Filter with a Constant Velocity (CV) model.
- *State Vector:* $x_k = [c_x, c_y, v_x, v_y]^T$, representing the centroid position and velocity.
- *Prediction:* The filter predicts the next state using the physical motion model:
  $ hat(x)_(k|k-1) = A hat(x)_(k-1|k-1) $
  $ P_(k|k-1) = A P_(k-1|k-1) A^T + Q $
- *Update:* Upon receiving a detection $z_k$ (centroid), the filter corrects its prediction:
  $ K_k = P_(k|k-1) H^T (H P_(k|k-1) H^T + R)^(-1) $
  $ hat(x)_(k|k) = hat(x)_(k|k-1) + K_k (z_k - H hat(x)_(k|k-1)) $

== System Diagram
#figure(
  diagram(
    node-stroke: 1pt,
    spacing: (1.5cm, 1.5cm),
    node((0,0), [Frame], shape: rect),
    edge("->"),
    node((1,0), [Detector], shape: rect, fill: yellow.lighten(80%)),
    edge("->"),
    node((2,0), [Measurement $z_k$], shape: rect),
    edge("->"),
    node((2,1), [Kalman Update], shape: rect, fill: green.lighten(80%)),
    edge((2,1), (1,1), "->"),
    node((1,1), [State Estimate $hat(x)_k$], shape: rect),
    edge((1,1), (0,1), "->"),
    node((0,1), [Kalman Predict], shape: rect, fill: blue.lighten(80%)),
    edge((0,1), (2,1), "->", [Prior $hat(x)_(k|k-1)$], label-pos: 0.5, label-side: center),
  ),
  caption: [Kalman Filter predict-update cycle.]
)

== Real Data Testing
We tested the tracker on `randomball.avi`.
- *Performance:* The filter successfully smooths the noisy detections of the ball.
- *Occlusion:* When the ball is momentarily not detected, the prediction step allows the tracker to maintain a trajectory estimate, although it drifts if the signal is lost for too long.

#figure(
  rect(width: 100%, height: 4cm, fill: luma(240))[
    #align(center + horizon)[*Placeholder: Screenshots of TP1 Tracking (Green: Det, Red: Pred)*]
  ],
  caption: [Kalman filter tracking results on the ball sequence.]
)

= TP 2: IoU-based Tracking (Bounding-Box Tracker)
== Objective
Develop a basic Multi-Object Tracker (MOT) using Intersection over Union (IoU) as the sole metric for data association.

== Methodology
This approach introduces track management:
1.  *Association:* We compute an IoU matrix between existing tracks (last known position) and new detections.
2.  *Hungarian Algorithm:* We maximize the total IoU to assign detections to tracks.
3.  *Lifecycle:*
    -   *Unmatched Detections* $arrow$ New Tracks.
    -   *Unmatched Tracks* $arrow$ Incremented "missed" counter; deleted if threshold exceeded.

== System Diagram
#figure(
  diagram(
    spacing: (1cm, 1.5cm),
    node-stroke: 1pt,
    node((0,0), [Detections], shape: rect),
    node((2,0), [Active Tracks], shape: rect),
    edge((0,0), (1,1), "->"),
    edge((2,0), (1,1), "->"),
    node((1,1), [IoU Matrix], shape: rect, fill: blue.lighten(90%)),
    edge("->"),
    node((1,2), [Hungarian Algo], shape: rect, fill: orange.lighten(80%)),
    edge("->"),
    node((1,3), [Track Management], shape: rect),
    edge((1,3), (0,3), "->", [New Tracks]),
    edge((1,3), (2,3), "->", [Update Tracks]),
    edge((1,3), (1,4), "->", [Delete Dead Tracks]),
  ),
  caption: [IoU-based association pipeline.]
)

== Real Data Testing
- *Success:* Works surprisingly well for pedestrians moving in distinct lanes without overlap.
- *Failure Cases:* The main limitation is the *Identity Switch*. When two pedestrians cross paths, their bounding boxes overlap significantly or one occludes the other. The IoU metric fails here, often swapping IDs or losing the occluded track.

#figure(
  rect(width: 100%, height: 4cm, fill: luma(240))[
    #align(center + horizon)[*Placeholder: TP2 Result - ID Switching Example*]
  ],
  caption: [Identity switch occurring during a crossing event in IoU tracker.]
)

= TP 3: Kalman-Guided IoU Tracking
== Objective
Improve the IoU tracker by predicting the future bounding box location using a Kalman Filter for each track before association.

== Methodology
Instead of comparing detections to the *last known* position of a track, we compare them to the *predicted* position.
- Each track has its own Kalman Filter ($x, y, v_x, v_y$).
- *Prediction:* Before association, `track.predict()` moves the bounding box to where the object is expected to be.
- *Association:* IoU is calculated between `predicted_box` and `detection_box`.

This allows the association gate to "follow" the object, making it robust to faster movements where the IoU between frame $t$ and $t-1$ might be low.

== System Diagram
#figure(
  diagram(
    spacing: 1.5cm,
    node-stroke: 1pt,
    node((0,0), [Tracks $(t-1)$], shape: rect),
    edge("->"),
    node((1,0), [Kalman Predict], shape: rect, fill: purple.lighten(80%)),
    edge("->"),
    node((2,0), [Predicted Boxes], shape: rect),
    
    node((2,1), [Detections $(t)$], shape: rect),
    edge((2,0), (2,0.5), (3, 0.5), (3, 1.5), (2.5, 1.5), "->"),
    edge((2,1), (2.5, 1.5), "->"),
    
    node((2.5, 1.5), [IoU Association], shape: rect, fill: blue.lighten(90%)),
    edge("->"),
    node((2.5, 2.5), [Kalman Update], shape: rect, fill: green.lighten(80%)),
  ),
  caption: [Kalman-guided tracking: Prediction precedes association.]
)

== Real Data Testing
- *Improvements:* Tracking is smoother. The system handles brief occlusions better because the Kalman filter continues to predict motion, maintaining a valid search region for re-association.
- *Limitations:* It still relies purely on geometry. If two objects have similar predicted positions (e.g., walking close together), the IoU metric alone cannot distinguish them.

#figure(
  rect(width: 100%, height: 4cm, fill: luma(240))[
    #align(center + horizon)[*Placeholder: TP3 Result - Improved Tracking continuity*]
  ],
  caption: [Kalman-Guided tracker maintaining ID through linear motion.]
)

= TP 4: Appearance-Aware IoU-Kalman Tracker
== Objective
Integrate visual features (Re-Identification or ReID) to resolve identity switches and handle long-term occlusions where geometric prediction becomes unreliable.

== Methodology
We utilize an *OSNet* model to extract a 512-dimensional feature vector for each detection crop.
- *Combined Cost:* The association cost matrix is a weighted sum:
  $ C_(i j) = alpha dot (1 - "IoU"_(i j)) + beta dot "CosineDist"("Feat"_i, "Feat"_j) $
- *Workflow:*
    1.  Predict track positions (Kalman).
    2.  Extract visual features from current frame detections.
    3.  Compute Combined Cost Matrix.
    4.  Solve assignment (Hungarian).
    5.  Update Kalman states and visual feature memory (moving average).

== System Diagram
#figure(
  diagram(
    spacing: (1cm, 1cm),
    node-stroke: 1pt,
    node((0,0), [Frame Crop], shape: rect),
    edge("->"),
    node((1,0), [OSNet ReID], shape: rect, fill: red.lighten(80%)),
    edge("->"),
    node((2,0), [Feature Vector], shape: rect),
    
    node((0,1), [Geometry (IoU)], shape: rect),
    
    node((1.5, 1.5), [Weighted Cost], shape: diamond, fill: yellow.lighten(80%)),
    edge((2,0), (1.5, 1.5), "->"),
    edge((0,1), (1.5, 1.5), "->"),
    
    edge((1.5, 1.5), (1.5, 2.5), "->"),
    node((1.5, 2.5), [Hungarian Match], shape: rect)
  ),
  caption: [Appearance-aware association fusing geometry and deep features.]
)

== Real Data Testing and Conclusion
- *Robustness:* This tracker is the most robust. In the `ADL-Rundle-6` sequence, we observed that even after a full occlusion behind a pillar/object, the tracker could re-identify the person correctly upon reappearance because their visual appearance hadn't changed.
- *Trade-off:* The inference speed is significantly lower due to the heavy CNN forward pass for every detection crop.
- *Conclusion:* While geometric tracking (TP3) is sufficient for simple, sparse scenarios, appearance information (TP4) is critical for crowded, complex environments where trajectories intersect.

#figure(
  rect(width: 100%, height: 4cm, fill: luma(240))[
    #align(center + horizon)[*Placeholder: TP4 Result - ReID recovering ID after occlusion*]
  ],
  caption: [Successful re-identification after occlusion using visual features.]
)

#bibliography("refs.bib")