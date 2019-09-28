import argparse
import cv2
from cv import VisionTargetDetector

# "python test.py 0" to run from camera in port 0
# "python test.py video.mp4" to run from the video recording video.mp4

parser = argparse.ArgumentParser()
parser.add_argument("input", help="read from the given camera or file", default="0")
args = parser.parse_args()

print("reading from", args.input)

vtd = VisionTargetDetector(args.input)
wait_time = 0 # wait time of 0 will wait indefinitely for next key press

# if input is from a camera, set wait time to 3 ms
if len(args.input) == 1:
	wait_time = 3

with vtd as v:
	while True:
		v.run_cv()
		key = cv2.waitKey(wait_time)
		if key == ord('q'):
			break
