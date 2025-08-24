#!/usr/bin/env python3
"""
multi_realsense_sync.py
Grab synchronized frames from *all* attached RealSense cameras and save them with wall-clock timestamps.
"""


"""

    target_type: 'aprilgrid'
    tagCols: 4
    tagRows: 3
    tagSize: 0.05      #size of one chessboard square [m]
    tagSpacing: 0.12      #size of one chessboard square [m]

FOLDER=$(pwd)
xhost +local:root
docker run -it -e DISPLAY -e QT_X11_NO_MITSHM=1 \
    --mount type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix,readonly=false \
    --mount type=bind,source="$FOLDER",target=/data \
    kalibr


source devel/setup.bash
rosrun kalibr kalibr_bagcreater --folder /data --output-bag calib_0.bag

rosrun kalibr kalibr_calibrate_cameras \    
    --bag /catkin_ws/calib_0.bag --target /data/target.yaml \
    --models pinhole-radtan pinhole-radtan pinhole-radtan pinhole-radtan\
    --topics /cam_0/image_raw /cam_1/image_raw /cam_2/image_raw /cam_3/image_raw --dont-show-report --export-poses > calib_log.txt
    
    
rosrun kalibr kalibr_calibrate_cameras \
    --bag /catkin_ws/calib_0.bag --target /data/target.yaml \
    --models pinhole-radtan pinhole-radtan pinhole-radtan pinhole-radtan pinhole-radtan\
    --topics /cam_0/image_raw /cam_1/image_raw /cam_2/image_raw /cam_3/image_raw /cam_4/image_raw --dont-show-report --approx-sync 0.03

rosrun kalibr kalibr_calibrate_cameras \
    --bag /catkin_ws/calib_0.bag --target /data/target.yaml \
    --models pinhole-radtan pinhole-radtan pinhole-radtan pinhole-radtan pinhole-radtan\
    --topics /cam_0/image_raw /cam_1/image_raw /cam_2/image_raw /cam_3/image_raw /cam_4/image_raw --verbose

kalibr_create_target_pdf --type apriltag --nx 6 --ny 6 --tsize 0.1 --tspace 0.2

Reinitialize the intrinsics for camera 0
        Projection initialized to: [915.89415999 913.88219114 641.35127548 336.59383398]
        Distortion initialized to: [ 0.10561959 -0.18864313 -0.00076718 -0.00028254]
Reinitialize the intrinsics for camera 1
        Projection initialized to: [916.514681   913.39620921 647.60912533 331.39301058]
        Distortion initialized to: [ 0.1126563  -0.19532986 -0.00575568  0.00168837]
Reinitialize the intrinsics for camera 2
        Projection initialized to: [909.07542175 907.78784989 605.4890762  345.1461511 ]
        Distortion initialized to: [ 0.10673813 -0.16098762 -0.00134143 -0.01000291]
Reinitialize the intrinsics for camera 3
        Projection initialized to: [904.26870238 901.89321286 595.74328546 378.36612332]
        Distortion initialized to: [ 0.10847777 -0.16780263  0.00319649 -0.01589969]
initializing initial guesses
         initializing camera pair (0,3)...  
         initializing camera pair (1,3)...  
         initializing camera pair (2,3)...  
initialized baseline between cam0 and cam1 to:
[[ 0.09208637 -0.97809424 -0.18668628  1.38346171]
 [ 0.44844445  0.20813019 -0.8692407   1.30428768]
 [ 0.88905437 -0.0036732   0.45778689 -0.06731906]
 [ 0.          0.          0.          1.        ]]
initialized baseline between cam1 and cam2 to:
[[ 0.20980344 -0.55329983  0.80612767 -1.07592117]
 [ 0.96089365  0.26908628 -0.06539089  2.54139903]
 [-0.18073713  0.78832219  0.58811752  0.78679169]
 [ 0.          0.          0.          1.        ]]
initialized baseline between cam2 and cam3 to:
[[ 0.96339183  0.2652072  -0.03925965 -0.65945717]
 [ 0.21995779 -0.69816115  0.68131459  2.68225463]
 [ 0.15327997 -0.66500838 -0.73093714  5.61331212]
 [ 0.          0.          0.          1.        ]]
initialized cam0 to:
"""

