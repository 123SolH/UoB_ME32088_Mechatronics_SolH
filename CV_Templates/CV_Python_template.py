"""
This script captures camera frames, displays color and grayscale images, detects ArUco markers,
estimates their pose, and reports the frame rate.

@author: ioannisgeorgilas
"""
import socket
import struct
import cv2
import logging
import numpy as np
import time
import cv2.aruco as aruco
logging.basicConfig(level=logging.INFO)

import os

# --- Laser Targeting Thresholds ---
ERROR_THRESHOLD_MM = 40.0  # Easily change this value here (+/- mm)
DESIRED_ID = 35  # Change your target marker ID here
# ----------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
calib_path = os.path.join(script_dir, 'Sample_Calibration.npz')

camera_calibration = np.load(calib_path)
CM = camera_calibration['CM']
dist_coef = camera_calibration['dist_coef']

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logging.error("Could not open camera 0")
    raise SystemExit(1)
logging.info("Camera 0 opened")

cap.set(3, 1920)
cap.set(4, 1080)
logging.info(
    "Camera resolution: %dx%d",
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
)

cv2.namedWindow("frame-image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("gray-image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("canny-image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("aruco-image", cv2.WINDOW_AUTOSIZE)

cv2.moveWindow("frame-image", 0, 100)
cv2.moveWindow("gray-image", 640, 100)
cv2.moveWindow("canny-image", 1280, 100)

marker_size = 40  # mm
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

half = marker_size / 2
marker_points = np.array([[-half, half, 0],
                          [half, half, 0],
                          [half, -half, 0],
                          [-half, -half, 0]], dtype=np.float32)

last_fps_log = time.perf_counter()

PI_IP = "138.38.229.206"  # Raspberry Pi IP address
PI_PORT = 9000
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:

  start = time.perf_counter()

  ret, frame = cap.read()
  if not ret:
    logging.error("Could not read a frame from camera 0")
    break

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  canny = cv2.Canny(frame, 50, 250)
  corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
  out = aruco.drawDetectedMarkers(frame, corners, ids)

  # Default values for every frame (assumes no fire unless conditions are met)
  laser_command = 0
  marker_id = -1
  x_err = 0.0
  y_err = 0.0

  # ids is None when no marker is visible, so only estimate pose if one was found
  if ids is not None and len(corners) > 0:
    ok, rvec, tvec = cv2.solvePnP(
        marker_points, corners[0], CM, dist_coef, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if ok:
      out = cv2.drawFrameAxes(out, CM, dist_coef, rvec, tvec, 10)

      marker_id = int(ids.flatten()[0])
      x_err = tvec[0][0]
      y_err = tvec[1][0]

      # Check if within error bounds and matches desired ID
      if (
          -ERROR_THRESHOLD_MM < x_err < ERROR_THRESHOLD_MM
          and -ERROR_THRESHOLD_MM < y_err < ERROR_THRESHOLD_MM
          and marker_id == DESIRED_ID
      ):
        laser_command = 1  # Fire laser

  # Pack and send the 1-byte command EVERY frame (sends 0 by default, 1 when targeted)
  packet = struct.pack("b", laser_command)
  udp_sock.sendto(packet, (PI_IP, PI_PORT))

  print(
      f"ID: {marker_id} | x: {x_err:.1f}, y: {y_err:.1f} |"
      f" Threshold: ±{ERROR_THRESHOLD_MM}mm | Command: {laser_command}"
  )

  cv2.imshow("frame-image", frame)
  cv2.imshow("gray-image", gray)
  cv2.imshow("canny-image", canny)
  cv2.imshow("aruco-image", out)

  end = time.perf_counter()

  if end - last_fps_log >= 1.0:
    logging.info("Camera FPS: %.2f", 1 / (end - start))
    last_fps_log = end

  if cv2.waitKey(20) & 0xFF == ord("q"):
    break



    cv2.imshow('frame-image', frame)
    cv2.imshow('gray-image', gray)
    cv2.imshow('canny-image', canny)
    cv2.imshow('aruco-image', out)

    end = time.perf_counter()

    if end - last_fps_log >= 1.0:
        logging.info("Camera FPS: %.2f", 1 / (end - start))
        last_fps_log = end

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
logging.info("Camera capture stopped")
exit(0)