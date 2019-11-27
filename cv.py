import math
import cv2
import numpy as np
import subprocess
import imghdr
import traceback
import os
from math import sin, cos

# finds angle between robot's heading and the perpendicular to the targets
class VisionTargetDetector:

    # initilaze variables
    def __init__(self, input):

        self.input_path = input

        try:
            # if input is a camera port
            self.input = cv2.VideoCapture(int(input))
            self.set_camera_settings(input)
        except:
            # if input is a path
            self.input = cv2.VideoCapture(input)

        frame = self.get_frame()

        # height of a vision target
        self.TARGET_HEIGHT = 5.5 * math.cos(math.radians(14.5)) + 2 * math.sin(math.radians(14.5))

        # intialize screen width and screen height
        self.SCREEN_HEIGHT, self.SCREEN_WIDTH = frame.shape[:2]

        # intialize angle of field of view in radians
        self.FIELD_OF_VIEW_RAD = 70.42 * math.pi / 180.0

        # calculates focal length based on a right triangle representing the "image" side of a pinhole camera
        # ABC where A is FIELD_OF_VIEW_RAD/2, a is SCREEN_WIDTH/2, and b is the focal length
        self.FOCAL_LENGTH_PIXELS = (self.SCREEN_WIDTH / 2.0) / math.tan(self.FIELD_OF_VIEW_RAD / 2.0)

        self.OBJ_POINT_SCALE = self.FOCAL_LENGTH_PIXELS * 5.5 / 24

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.input.release()
        cv2.destroyAllWindows()
        print("\nexited")

    # sets exposure of the camera (will only work on Linux systems)
    def set_camera_settings(self, camera_port):

        camera_path = "/dev/video" + camera_port

        try:
            subprocess.call(["v4l2-ctl", "-d", camera_path, "-c", "exposure_auto=1"])
            subprocess.call(["v4l2-ctl", "-d", camera_path, "-c", "exposure_absolute=1"])
        except:
            print("exposure adjustment not completed")

    # returns a frame corresponding to the input type
    def get_frame(self):

        frame = None

        try:
            # if input is an image, use cv2.imread()
            if imghdr.what(self.input_path) is not None:
                frame = cv2.imread(self.input_path)
            # if input is a video, use VideoCapture()
            else:
                _, frame = self.input.read()
        except:
            # if input is a camera port, use VideoCapture()
            _, frame = self.input.read()

        return frame

    # returns the closest pair of vision targets
    def get_closest_pair(self, pairs):

        if len(pairs) == 0:
            return None

        closest_pair = pairs[int(len(pairs)/2)]

        for pair in pairs:
            if abs(self.SCREEN_WIDTH/2 - pair.get_center().x) < abs(self.SCREEN_WIDTH/2 - closest_pair.get_center().x):
                closest_pair = pair

        return closest_pair

    # returns an array of all vision target pairs
    def get_all_pairs(self, rotated_boxes):

        pairs = []

        for c in range(0, len(rotated_boxes)-1):
            rect1, rect2 = rotated_boxes[c], rotated_boxes[c+1]
            top_distance = math.hypot(rect1.p2.x - rect2.p2.x, rect1.p2.y - rect2.p2.y)
            bottom_distance = math.hypot(rect1.p4.x - rect2.p4.x, rect1.p4.y - rect2.p4.y)

            if top_distance < bottom_distance:
                pairs.append(Pair(rect1, rect2, self))

        return pairs

    def get_euler_from_rodrigues(self, rmat):
        sin = math.sqrt(rmat[2][0] * rmat[2][0] + rmat[2][1] * rmat[2][1])
        if sin >= 1e-6:
            z1 = math.atan2(rmat[2][0], rmat[2][1])  # around z1-axis
            x = math.atan2(sin, rmat[2][2])  # around x-axis
            z2 = math.atan2(rmat[0][2], -rmat[1][2])  # around z2-axis
        else:  # gimbal lock
            z1 = 0  # around z1-axis
            x = math.atan2(sin, rmat[2][2])  # around x-axis
            z2 = 0  # around z2-axis

        euler = np.array([[z1], [x], [z2]])

        euler_deg = -180 * euler / math.pi
        # euler_deg[0][0] = (360 - euler_deg[0][0])%360
        euler_deg[1][0] = euler_deg[1][0]

        return euler_deg[0][0], euler_deg[1][0], euler_deg[2][0]

    def get_angle_dist(self, pair):

        w = self.SCREEN_WIDTH
        h = self.SCREEN_HEIGHT
        x = math.radians(14.5)

        # k = self.OBJ_POINT_SCALE / math.hypot(2, 5.5)
        k = 9

        # array of object points(3D points)
        obj_points = []
        obj_points.append(k*np.array([w/(2*k), h/(2*k), 0]))
        obj_points.append(k*np.array([w/(2*k) - 4 - 2*cos(x), 0.5*(h/k - 5.5*cos(x) - 2*sin(x)), 0])) #A
        obj_points.append(k*np.array([w/(2*k) - 4, 0.5*(h/k - 5.5*cos(x) + 2*sin(x)), 0])) #B
        obj_points.append(k*np.array([w/(2*k) - 4 - 2*cos(x) - 5.5*sin(x), 0.5*(h/k + 5.5*cos(x) - 2*sin(x)), 0])) #C
        obj_points.append(k*np.array([w/(2*k) - 4 - 5.5*sin(x), 0.5*(h/k + 5.5*cos(x) + 2*sin(x)), 0])) #D
        obj_points.append(k*np.array([w/(2*k) + 4 + 2*cos(x), 0.5*(h/k - 5.5*cos(x) - 2*sin(x)), 0])) #Q
        obj_points.append(k*np.array([w/(2*k) + 4, 0.5*(h/k - 5.5*cos(x) + 2*sin(x)), 0])) #R
        obj_points.append(k*np.array([w/(2*k) + 4 + 5.5*sin(x) + 2*cos(x), 0.5*(h/k + 5.5*cos(x) - 2*sin(x)), 0])) #S
        obj_points.append(k*np.array([w/(2*k) + 4 + 5.5*sin(x), 0.5*(h/k + 5.5*cos(x) + 2*sin(x)), 0]))

        #Array of image points(2D points)
        img_points = []
        img_points.append([pair.get_center().x, pair.get_center().y])
        for p in pair.left_rect.points:
            img_points.append([p.x,p.y])
        for p in pair.right_rect.points:
            img_points.append([p.x,p.y])

        frame = self.get_frame()
        for p in img_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 5, (255,0,255), 5)

        cv2.imshow("points", frame)

        # img_points = []
        # for p in self.obj_points:
        #   img_points.append([2*p[0] + 160, 2*p[1]])

        #Convert the object and image point arrays into numpy arrays so they can be passed into solvePnP
        obj_points = np.float64(obj_points)
        # img_points = np.float64(obj_points)
        img_points = np.float64(img_points)

        #Camera matrix
        camera_matrix = np.float64([[self.FOCAL_LENGTH_PIXELS,0,self.SCREEN_WIDTH/2],
                                  [0,self.FOCAL_LENGTH_PIXELS,self.SCREEN_HEIGHT/2],
                                  [0,0,1]])

        #Returning solvePnP which returns the rotation vector and translation vector
        bool, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, None)
        return rvec, tvec, np.int32(obj_points)

    def get_corners(self, contour):

        hull = cv2.convexHull(contour)
        hull_changed = []
        for i in range(len(hull)):
            hull_changed.append([hull[i][0][0], hull[i][0][1]])

        max = 0
        max_arr = []
        for i in range(len(hull_changed)):
            for j in range(i):
                for k in range(j):
                    for m in range(k):
                        total = 0
                        total += math.hypot(hull_changed[i][0]-hull_changed[j][0],hull_changed[i][1]-hull_changed[j][1])
                        total += math.hypot(hull_changed[j][0]-hull_changed[k][0],hull_changed[j][1]-hull_changed[k][1])
                        total += math.hypot(hull_changed[k][0]-hull_changed[m][0],hull_changed[k][1]-hull_changed[m][1])
                        total += math.hypot(hull_changed[m][0]-hull_changed[i][0],hull_changed[m][1]-hull_changed[i][1])
                        if(total>max):
                            max = total
                            max_arr = [hull_changed[i],hull_changed[j],hull_changed[k],hull_changed[m]]

        arrmax_changed = []
        for i in range(len(max_arr)):
            arrmax_changed.append([max_arr[i]])
        return np.int0(arrmax_changed)

    def run_cv(self):

        frame = self.get_frame()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        low_green = np.array([65,145,65])
        high_green= np.array([87,255,229])

        # isolate the desired shades of green
        mask = cv2.inRange(hsv, low_green, high_green)

        if (self.input_path.isdigit()):
            _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        # sort contours by x-coordinate
        contours.sort(key = lambda countour: cv2.boundingRect(countour)[0])

        rotated_boxes = []

        # convert contours into rectangles
        for c in contours:
            area = cv2.contourArea(c)
            rect = cv2.minAreaRect(c)
            _, _, rot_angle = rect
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            if area > 100:
                corners = self.get_corners(c)
                if len(corners) > 3:
                    rotated_boxes.append(RotatedRectangle(corners, area, rot_angle))
                    cv2.drawContours(frame, corners, -1, (0,0,255), 2)
                else:
                    cv2.drawContours(frame, corners, -1, (0,255,255), 2)

        pair = self.get_closest_pair(self.get_all_pairs(rotated_boxes))

        if pair is None:
            cv2.imshow("contours: " + str(self.input_path), mask)
            cv2.imshow("frame: " + str(self.input_path), frame)
            return [-999, -999, -999], [0, 0, 0]

        cv2.drawContours(frame, [pair.left_rect.box], 0, (255,0,0), 1)
        cv2.drawContours(frame, [pair.right_rect.box], 0, (255,0,0), 1)
        r, t, o = self.get_angle_dist(pair)
        rmat, _ = cv2.Rodrigues(r)

        yaw, pitch, roll = self.get_euler_from_rodrigues(rmat)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Yaw: " + str(round(yaw,2)), (20, self.SCREEN_HEIGHT - 90), font, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Pitch: " + str(round(pitch,2)), (20, self.SCREEN_HEIGHT - 60), font, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Roll: " + str(round(roll,2)), (20, self.SCREEN_HEIGHT - 30), font, 1, (255,255,255), 2, cv2.LINE_AA)
        # print("\nYaw: {}\nPitch: {}\nRoll: {}\n".format(yaw, pitch, roll))
        # print("Translation vector: \n", t)

        for i in o:
            cv2.circle(frame, (i[0], i[1]), 1, (225,0,0), 5)

        # show windows
        cv2.imshow("contours: " + str(self.input_path), mask)
        cv2.imshow("frame: " + str(self.input_path), frame)

        return [yaw, pitch, roll], t

# this class defines the bounding rectangle of a vision target
class RotatedRectangle:

    def __init__(self, box, area, rot_angle):
        self.box = box
        self.area = area
        self.rot_angle = rot_angle

        points = []
        for coordinates in box:
            points.append(Point(coordinates[0][0], coordinates[0][1]))

        # sorts points based on y value
        points.sort(key = lambda x: x.y)

        # highest = 1, lowest = 4
        self.points = points
        self.p1, self.p2, self.p3, self.p4 = points[0], points[1], points[2], points[3]

    def get_width(self):
        return math.hypot(self.p1.x - self.p2.x, self.p1.y - self.p2.y)

    def get_height(self):
        return math.hypot(self.p1.x - self.p3.x, self.p1.y - self.p3.y)

    def get_center(self):
        x = sum(point.x for point in self.points)/4
        y = sum(point.y for point in self.points)/4
        return Point(x, y)

# this class defines a point
class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_coordinate(self):
        return [self.x, self.y]

# this class defines a pair of vision targets
class Pair:
    def __init__(self, left_rect, right_rect, vtd):
        self.left_rect = left_rect
        self.right_rect= right_rect
        self.vtd = vtd

    def get_center(self):
        r1 = self.left_rect
        r2 = self.right_rect
        x = (self.left_rect.get_center().x + self.right_rect.get_center().x)/2
        y = (self.left_rect.get_center().y + self.right_rect.get_center().y)/2
        return Point(x, y)
