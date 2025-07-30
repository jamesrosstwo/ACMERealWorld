import cv2
from pupil_apriltags import Detector

# Initialize the detector
at_detector = Detector(
   families="tag36h11",
   nthreads=1,
   quad_decimate=1.0,
   quad_sigma=0.0,
   refine_edges=1,
   decode_sharpening=0.25,
   debug=0
)

# Load image in grayscale
image_path = r"C:\Users\User\Desktop\ACMERealWorld\outputs\kalibr_postprocess_20250728_010634\cam3\1753720185340389120.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Failed to load image. Check the file path.")
else:
    # Detect AprilTags
    detections = at_detector.detect(img)
    print(f"Detected {len(detections)} tags")

    # Convert grayscale to BGR color for visualization
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for detection in detections:
        # Each detection has corners as a list of four (x, y) tuples in order:
        # [top-left, top-right, bottom-right, bottom-left]
        corners = detection.corners.astype(int)

        # Draw the outline of the tag (a polygon)
        for i in range(4):
            pt1 = tuple(corners[i])
            pt2 = tuple(corners[(i + 1) % 4])
            cv2.line(img_color, pt1, pt2, (0, 255, 0), 2)

        # Draw the tag ID near the top-left corner
        tag_id = detection.tag_id
        cv2.putText(img_color, str(tag_id), tuple(corners[0] - [0, 10]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the image with detections
    cv2.imshow("AprilTag Detections", img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
