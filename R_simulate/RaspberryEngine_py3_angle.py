#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'duc_tin'

import cv2
import numpy as np
import time
from wsgiref.simple_server import make_server
from urllib.parse import parse_qs
from collections import deque
from threading import Thread


def rotate_img(image, angle):
    rows, cols = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
    return cv2.warpAffine(image, rot_mat, (cols, rows), flags=cv2.INTER_LINEAR)


def midPoint(xxx_todo_changeme, xxx_todo_changeme1):
    """find middle point of segment AB"""
    (xA, yA) = xxx_todo_changeme
    (xB, yB) = xxx_todo_changeme1
    xA, yA = float(xA), float(yA)
    xB, yB = float(xB), float(yB)
    return np.array([(xA + xB) / 2, (yA + yB) / 2], dtype=int)


def cordShift(xxx_todo_changeme2):
    """Shift the XY origin from top left into middle of the screen"""
    (X, Y) = xxx_todo_changeme2
    return np.array((X - origin[0], Y - origin[1]))


def cm2pixel(Varlist):
    """Convert centimet to pixel"""
    k = 37.79527
    var = np.array(Varlist) * k
    return var.astype('int')


def earthCord(Varlist):
    """ Convert screen coordinate (X,Y) into relative ground coordinate (x,y)"""
    retlist = []

    for (WX, WY) in Varlist:
        # X, Y = np.absolute(cordShift((WX, WY)))
        X, Y = cordShift((WX, WY))
        beta = np.arctan(Y / k)
        # if (WY - 239) < 0: beta = -beta

        AWY = np.sqrt(k ** 2 + Y ** 2)
        AWy = h / np.cos(np.pi / 2 - alpha - beta)
        OWy = np.sqrt(AWy ** 2 - h ** 2)
        Wx = X * AWy / AWY
        retlist.append((Wx, OWy))
    return np.array(retlist)


class Simulation():
    def __init__(self, course, start_at, speed, real_view_zone):
        self.course = course
        self.pos = start_at  # set start position at
        self.speed = speed  # pixel/frame
        self.angle = 0  # angle start at y axis (sin axis) and clockwise
        self.noWarp = 0  # the view of camera before warp
        self.rot_view_shift = None
        self.rvz = real_view_zone

        # constant for warping 2D -> 3D
        # self.hdelta = delta / 2
        # self.width = int(3.2 * delta)
        # bottom_left, bottom_right, up_right, up_left
        # (-283, 420) (283, 420) (1290, 2758) (-1290, 2758)
        # (1007, 2338) (1573, 2338) (2580, 0) (0, 0)
        # (0, 0) (1007, 2338) (1573, 2338) (2580, 0)
        # (3181, 4479)

        rvz = self.rvz
        rvz[:, 0] -= rvz[0, 0]
        rvz[:, 1] = rvz[0, 1] - rvz[:, 1]

        self.view_matrix = rvz / cooef
        # self.view_matrix = np.array([[0, 0], [1007, 2338], [1573, 2338], [2580, 0]]) / cooef
        self.midpoint = -midPoint(*self.view_matrix[1:3]) + (delta, delta)
        self.view_shift = self.view_matrix + self.midpoint
        self.mask = np.zeros((int(2 * delta), int(2 * delta)), np.uint8)  ###int(2*delta)
        cv2.drawContours(self.mask, self.view_shift.reshape((1, 4, 2)).astype(np.int32), 0, 255, -1)

        self.x0, self.x4 = int(self.view_shift[0][0]), int(self.view_shift[-1][0])
        self.y0, self.y1 = int(self.view_shift[0][1]), int(self.view_shift[1][1])

        self.ly, self.lx = self.y1 - self.y0, self.x4 - self.x0
        self.dy, self.dx = 3 * self.view_matrix[1, 0], self.view_matrix[1, 0]
        # top_left, bottom_left, bottom_right, top_right
        self.approx = np.array([[0, 0], [self.dx, self.ly], [self.lx - self.dx, self.ly], [self.lx, 0]], np.float32)
        self.standard = np.array([[0, 0], [0, self.ly], [self.lx, self.ly], [self.lx, 0]], np.float32)
        self.retval = cv2.getPerspectiveTransform(self.approx, self.standard)

    def move(self, command):
        """ move using complex command (複素数を入力とした走行指令)"""
        agl = np.deg2rad(self.angle)
        if type(command) == complex:
            command_vec = int(command.real), int(command.imag)
            if command_vec == (1, 0):  # 'forward'
                self.pos[0] -= self.speed * np.cos(agl)
                self.pos[1] += self.speed * np.sin(agl)
            elif command_vec == (-1, 0):  # 'backward'
                self.pos[0] += self.speed * np.cos(agl)
                self.pos[1] -= self.speed * np.sin(agl)
            elif command_vec == (0, 1):  # 'turn left':
                self.angle += command_vec[1]
                self.rotate(self.angle)
            elif command_vec == (0, -1):  # 'turn right':
                self.angle += command_vec[1]
                self.rotate(self.angle)
            else:
                print('arbitrary angle command is not implemented yet')
        else:
            if command[0] == 'forward':
                self.pos[0] -= self.speed * np.cos(agl)
                self.pos[1] += self.speed * np.sin(agl)
            elif command[0] == 'backward':
                self.pos[0] += self.speed * np.cos(agl)
                self.pos[1] -= self.speed * np.sin(agl)
            elif command[0] == 'turn left':
                self.angle -= command[1]
                self.rotate(self.angle)
            elif command[0] == 'turn right':
                self.angle += command[1]
                self.rotate(self.angle)

    def rotate(self, angle):
        if angle > 360 or angle < - 360:
            sign = -360 if angle < 0 else 360
            self.angle %= sign

    def camera(self):
        """Get the view from PI camera"""
        agl = np.deg2rad(self.angle)
        cos, sin = np.cos(agl), np.sin(agl)
        self.rot_view_shift = np.array([[x * cos - y * sin, x * sin + y * cos] for x, y in self.view_matrix], np.int64)
        self.rot_view_shift += (self.pos[::-1] - midPoint(*self.rot_view_shift[1:3])).astype(self.rot_view_shift.dtype)

        # take a smaller image which covers the view zone
        cen_y, cen_x = int(self.pos[0]), int(self.pos[1])
        x_min, x_max = cen_x - delta, cen_x + delta
        y_min, y_max = cen_y - delta, cen_y + delta
        #print('(xmin,xmax,ymin,ymax)=', x_min.astype(np.int32), x_max.astype(np.int32), y_min.astype(np.int32), y_max.astype(np.int32))
        image = self.course[max(0, int(y_min)):min(self.course.shape[0], int(y_max)),
                max(0, int(x_min)):min(self.course.shape[1], int(x_max))]

        # in case one or more vertexes is out of image's size
        ground_mask = np.zeros((int(2 * delta), int(2 * delta), 3), np.uint8)
        dx_min = 0 - x_min if x_min < 0 else 0
        dy_min = 0 - y_min if y_min < 0 else 0

        if y_max > 0 and x_max > 0:
            ground_mask[int(dy_min):int(dy_min) + image.shape[0], int(dx_min):int(dx_min) + image.shape[1]] = image

        # rotate and take the view zone out
        rot_mask = rotate_img(ground_mask, self.angle)
        self.noWarp = cv2.bitwise_and(rot_mask, rot_mask, mask=self.mask)[self.y0:self.y1, self.x0:self.x4]
        # cv2.imshow("PI camera", self.noWarp)
        # cv2.waitKey()
        warp = cv2.warpPerspective(self.noWarp, self.retval, (self.lx, self.ly), flags=cv2.INTER_CUBIC)
        return cv2.resize(warp, (col, row), interpolation=cv2.INTER_CUBIC)


class HTTPdaemon:
    """Use this class to control real raspberry"""

    def __init__(self, image_deque):
        self.imqueue = image_deque
        self.cache = None

    def application(self, environ, start_response):
        # receive request from client
        query = parse_qs(environ['QUERY_STRING'])
        response_body = ' Sending Images over <a href="127.0.0.1:8888/?action=stream"> here</a> '  ###
        status = '200 OK'
        response_headers = [('Content-Type', 'text/html'),
                            ('Content-Length', str(len(response_body)))]

        if 'action' in query:
            response_body = self.feeder()
            response_headers = [('Content-type', 'image/jpeg'),
                                ('Content-Length', str(len(response_body)))]
            if query['action'][0] == 'stream':
                pass
                # response_body = self.feeder()
                # response_headers = [('Content-type', 'image/jpeg'),
                #                    ('Content-Length', str(len(response_body)))]

            elif query['action'][0] == 'forward':
                rasp_pi.move(('forward', 0))
            elif query['action'][0] == 'left':
                rasp_pi.move(('turn left', angle))
            elif query['action'][0] == 'right':
                rasp_pi.move(('turn right', angle))
            elif query['action'][0] == 'backward':
                rasp_pi.move(('backward', 0))

        if 'direction' in query:
            response_body = self.feeder()
            response_headers = [('Content-type', 'image/jpeg'),
                                ('Content-Length', str(len(response_body)))]
            motor1, motor2 = query['direction'][0].split()
            motor = complex(int(motor1), int(motor2))
            rasp_pi.move(motor)

        start_response(status, response_headers)

        return [response_body]

    def feeder(self):
        if self.imqueue:
            self.cache = self.imqueue.pop()

        return self.cache


# ------------------------- Main OS  ---------------------------------

# camera setting ============
alpha = np.deg2rad(30)  # angle between camera and plan xOy
h = 12.5  # cm, height of camera
k = 650  # pixel, distance from camera to virtual screen
AC = h / np.sin(alpha)
OC = h / np.tan(alpha)
row, col = 480, 640  # image size on the screen
origin = np.array((col / 2, row / 2), dtype=float)

# top_left, bottom_left, bottom_right, top_right
window = np.array([(0, 0), (0, 480), (640, 480), (640, 0)])
g_win_raw = (earthCord(window) * 37.79527).astype('int')
g_win = np.abs(g_win_raw - (min(g_win_raw[:, 0]), max(g_win_raw[:, 1])))

# pi setting: ===============
cooef = 4  # zooming coefficient: 1,2,4,...
pi_speed = 10  # pixel per frame
angle = 2  # degrees per frame, rotation speed
fps = 30  # frame rate, not exact that number due to time of functions call
delta = np.max(g_win) / cooef + 10  # pixel, half-width of the view zone from camera

# prepare window to show image
cv2.namedWindow("PI camera", 1)
# cv2.namedWindow("Ground", 2)

# load course and start simulating
course = cv2.imread("course001reduced.png")
print(type(course))
pi_pos = [course.shape[0] / 2, course.shape[1] / 2]  # middle or anywhere within image's border
rasp_pi = Simulation(course, start_at=pi_pos, speed=pi_speed, real_view_zone=g_win_raw)

# queue
imqueue = deque(maxlen=2)

# HTTPサーバのポート設定　============
ip, port = '127.0.0.1', 8888

# make us online forever in a different thread
app = HTTPdaemon(imqueue)
server = make_server(ip, port, app.application)
t = Thread(target=server.serve_forever)
t.daemon = True
t.start()

# cv2.namedWindow("Ground", 2)

t0 = time.time()
tmp = 0.
while True:
    # test frame rate
    tmp += 1
    dt = time.time() - t0
    if dt > 5:
        print('frame rate: %.2f / s' % (tmp / dt))
        # print rasp_pi.angle, rasp_pi.pos[1]-1589, rasp_pi.pos[0]-1123
        t0 = time.time()
        tmp = 0

    # get the view from PI camera
    view = rasp_pi.camera()
    # y, x = view.shape[:2]
    # cv2.circle(view, (x / 2, y / 2), 5, (0, 255, 128), -1)
    cv2.imshow("PI camera", view)

    # ground = rasp_pi.course.copy()
    # views = rasp_pi.rot_view_shift
    # cv2.drawContours(ground, views.reshape((1, 4, 2)), 0, 255, 3)
    # y, x = ground.shape[:2]
    # cv2.circle(ground, (x / 2, y / 2 + delta + 60), 5, (0, 255, 128), -1)
    # cv2.imshow("Ground", ground)

    # put image into deque to send over http
    ret, encoded = cv2.imencode('.jpg', view)
    imqueue.append(encoded.tostring())

    # receive command from keyboard or from other module　(キーボード入力)
    key = cv2.waitKey(int(1000 / fps))
    if key == -1:
        continue

    command = "", 0
    if key == ord('w'):
        command = 'forward', 0
    elif key == ord('s'):
        command = 'backward', 0
    elif key == ord('a'):
        command = 'turn left', angle
    elif key == ord('d'):
        command = 'turn right', angle
    elif key == 27:
        # Esc to exit
        print('Program finished')
        break

    # 走行
    rasp_pi.move(command)
