import socket
import time
import os
import subprocess
import cv2
import argparse
from datetime import datetime
from cv import VisionTargetDetector

# "python test.py 0" to run from camera in port 0
# "python test.py video.mp4" to run from the video recording video.mp4
# "python test.py --out output.mp4" to save output to a recording file

parser = argparse.ArgumentParser()
parser.add_argument("input", help="read from the given camera or video", default="0")
args = parser.parse_args()

print("Reading from", args.input)

# df = VisionTargetDetector("0")
# db = VisionTargetDetector("1")

vtd = VisionTargetDetector(args.input)

#with df, db as d1, d2:
while True:
    # angle, distance = d.run_cv()
    # print(df.run_cv())
    # print(db.run_cv())
	print(vtd.run_cv())

# close input and writer
detector.release_cv_objects()
