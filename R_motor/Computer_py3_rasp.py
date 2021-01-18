#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'duc_tin'

import numpy as np
import urllib.request, urllib.error, urllib.parse, requests
import cv2
import random
import time
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server
# from urlparse import urllib.parse
import urllib
from collections import deque
from threading import Thread
import json
import pprint
import csv

robot_pose = [0.0, 0.0, 0.0]
def send_cmd_reset():
    # ラズパイにデータ送信
    response = requests.get('http://192.168.179.8:8888/?reset=reset')
    response_list = response.text.split(',')
    robot_pose[0] = float(response_list[0])
    robot_pose[1] = float(response_list[1])
    robot_pose[2] = float(response_list[2])
    print("[info] sent reset cmd")

def send_cmd_direction(vdirect):
    # 機体速度の設定
    max_linear_speed = 255
    max_angular_speed = 300
    linear_speed = max_linear_speed * vdirect.real
    angular_speed = max_angular_speed * vdirect.imag

    # ホイール速度の算出
    cmd_wheel_speed = [0, 0]
    if angular_speed != 0 and linear_speed == 0:
        cmd_wheel_speed[0] = -angular_speed/2
        cmd_wheel_speed[1] = angular_speed/2
    else:
        cmd_wheel_speed[0] = linear_speed - np.sign(linear_speed) * angular_speed / 2
        cmd_wheel_speed[1] = linear_speed + np.sign(linear_speed) * angular_speed / 2

    # 出力のガード
    if abs(cmd_wheel_speed[0]) > 255 or abs(cmd_wheel_speed[1]) > 255:
        if abs(cmd_wheel_speed[0]) > abs(cmd_wheel_speed[1]):
            cmd_wheel_speed[1] = cmd_wheel_speed[1] * 255 / abs(cmd_wheel_speed[0])
            cmd_wheel_speed[0] = np.sign(cmd_wheel_speed[0]) * 255
        else:
            cmd_wheel_speed[0] = cmd_wheel_speed[0] * 255 / abs(cmd_wheel_speed[1])
            cmd_wheel_speed[1] = np.sign(cmd_wheel_speed[1]) * 255

    # float から int にする
    cmd_wheel_speed[0] = int(cmd_wheel_speed[0])
    cmd_wheel_speed[1] = int(cmd_wheel_speed[1])

    # ラズパイにデータ送信
    response = requests.get('http://192.168.179.8:8888/?action=' + str(cmd_wheel_speed[0]) + ',' + str(cmd_wheel_speed[1]))
    response_list = response.text.split(',')
    robot_pose[0] = float(response_list[0])
    robot_pose[1] = float(response_list[1])
    robot_pose[2] = float(response_list[2])
    print("[cmd_wheel_speed] L:" + str(cmd_wheel_speed[0]) + ", R:" + str(cmd_wheel_speed[1]))


class HTTPdaemon:
    """Use this class to control real raspberry"""

    def __init__(self, image_deque):
        self.imqueue = image_deque
        self.cache = None

    def application(self, environ, start_response):
        query = parse_qs(environ['QUERY_STRING'])

        if 'pose' in query:
            pose = query['pose'][0]
            pose_list = pose.split(',')
            print(pose_list)

        response_body = 'Getted Pose'  ###
        status = '200 OK'
        response_headers = [('Content-Type', 'text/html'),
                            ('Content-Length', str(len(response_body)))]
        start_response(status, response_headers)
        return [response_body.encode('utf-8')]


# ------ Main --------------
ip, port = '192.168.179.6', 8888
imqueue = deque(maxlen=2)
app = HTTPdaemon(imqueue)
server = make_server(ip, port, app.application)
t = Thread(target=server.serve_forever)
t.daemon = True
t.start()

# HTTPサーバのポート設定　============
ip, port = '192.168.179.8', 8080

# 画像取得　============
stream = urllib.request.urlopen('http://%s:%s/?action=stream' % (ip, port))
bytes = bytes(b'')
count = 0

# 指令の指定　============
pi_commands = ['forward', 'left', 'right', 'backward']
pi_commands_vec = [complex(1, 0), complex(0, 1), complex(0, -1), complex(-1, 0)]
Timeout = 2  # second
t0 = 0

start = time.time()

follow_start = False
moon_detected = False
save_map = False

seq_num = 0
seq_end_time = time.time()

loop_count = 0
pose_list = []

send_cmd_reset()

# main loop　============
while 1:
    bytes += stream.read(1024)
    a = bytes.find(b'\xff\xd8')
    b = bytes.find(b'\xff\xd9')
    # take video from stream
    if not (a != -1 and b != -1):
        continue

    # Simulation is kept as simple as possible, it's not a real
    # mjpeg stream but a single jpg image.
    # The below line only needed when receiving img from simulation
    # stream = urllib.request.urlopen('http://%s:%s/?action=stream' % (ip, port))

    jpg = bytes[a:b + 2]
    bytes = bytes[b + 2:]

    img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.flip(img, 0)
    cv2.imshow('received', img)

    if time.time() - start > 0.2:
        loop_count += 1
        key = cv2.waitKey(1)
        if key == 27:  # break when press ESC
            send_cmd_direction(complex(0.0, 0.0))
            break

        # ラジコン
        if key == 115:    # s: 停止
            send_cmd_direction(complex(0, 0))
        elif key == 119:    # w: 前
            send_cmd_direction(complex(1, 0))
        elif key == 120:    # x: 後
            send_cmd_direction(complex(-1, 0))
        elif key == 97:    # a: 左
            send_cmd_direction(complex(0, 1))
        elif key == 100:    # d: 右
            send_cmd_direction(complex(0, -1))
        elif key == 113:    # q: 左前
            send_cmd_direction(complex(1, 1))
        elif key == 101:    # e: 右前
            send_cmd_direction(complex(1, -1))
        elif key == 122:    # z: 左後
            send_cmd_direction(complex(-1, 1))
        elif key == 99:    # c: 右後
            send_cmd_direction(complex(-1, -1))
        elif key == 112:    # p: トレーススタート
            follow_start = True
        elif key == 109:    # m: 月発見
            moon_detected = True
        elif key == 108:    # l: map保存
            save_map = True

        start = time.time()
        # グレースケール化
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('img_gray', img_gray)

        # 射影変換
        # 0 1
        # 2 3
        p_original = np.float32([[0, 0], [img_gray.shape[1], 0],
                                [0 - 227*2, img_gray.shape[0]], [img_gray.shape[1] + 227*2, img_gray.shape[0]]])
        p_trans = np.float32([[0, 0], [img_gray.shape[1], 0],
                            [0, img_gray.shape[0] + 18*2], [img_gray.shape[1], img_gray.shape[0] + 18*2]])
        M = cv2.getPerspectiveTransform(p_original, p_trans)
        img_projection = cv2.warpPerspective(img_gray, M, (p_trans[3][0], p_trans[3][1]), borderValue=(255, 255, 255))
        cv2.imshow('img_projection', img_projection)

        # エッヂの検出
        img_edge = cv2.Canny(img_projection,10,100)
        # cv2.imshow('img_edge', img_edge)

        # # 2値化
        _, img_binarized = cv2.threshold(img_projection, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # cv2.imshow('img_binarized', img_binarized)

        # マーカー検出
        img_marker = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR) # 表示用画像

        moons = []
        cv_moons = cv2.HoughCircles(img_edge, cv2.HOUGH_GRADIENT, 1, 1000, param1=10, param2=15, minRadius=30, maxRadius=50)
        if cv_moons is not None:
            for i in cv_moons[0,:]:
                moons.append([i[0], i[1], i[2]])
                cv2.circle(img_marker, (i[0],i[1]), np.uint16(i[2]), (0,255,255), 2)    # 囲み線
                cv2.circle(img_marker, (i[0],i[1]), 2, (0,0,255), 3)                    # 中心点
            # print(moons)

        contours, _ = cv2.findContours(img_binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, cnt in enumerate(contours):
            # remove small objects
            if cv2.contourArea(contours[i]) < 500:
                continue

            rect = cnt
            x, y, w, h = cv2.boundingRect(rect)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 10)

            # print(cv2.contourArea(cnt))
            # print("s")
            # print(cnt)

        circles = []
        cv_circles = cv2.HoughCircles(img_edge, cv2.HOUGH_GRADIENT, 1, 40, param1=10, param2=12, minRadius=9, maxRadius=15)
        if cv_circles is not None:
            for i in cv_circles[0,:]:
                if len(moons) != 0:
                    # 月までの距離
                    dist_to_moon = np.sqrt((moons[0][0] - i[0])**2 + (moons[0][1] - i[1])**2)

                    if dist_to_moon > moons[0][2]:
                        circles.append([i[0], i[1], i[2]])
                        cv2.circle(img_marker, (i[0],i[1]), np.uint16(i[2]), (255,255,0), 2)    # 囲み線
                        cv2.circle(img_marker, (i[0],i[1]), 2, (0,0,255), 3)                    # 中心点
                else:
                    circles.append([i[0], i[1], i[2]])
                    cv2.circle(img_marker, (i[0],i[1]), np.uint16(i[2]), (255,255,0), 2)    # 囲み線
                    cv2.circle(img_marker, (i[0],i[1]), 2, (0,0,255), 3)                    # 中心点

            # print(circles)

        cv2.imshow('img_marker', img_marker)

        # 走行制御
        if moon_detected:
            # 月を発見
            if seq_num == 0:
                # 停止
                send_cmd_direction(complex(0.0, 0.0))
                seq_end_time = time.time()
                seq_num += 1
            elif seq_num == 1:
                # 待機
                send_cmd_direction(complex(0.0, 0.0))
                if time.time() - seq_end_time > 1:
                    seq_end_time = time.time()
                    seq_num += 1
            elif seq_num == 2:
                # 旋回
                send_cmd_direction(complex(0.0, 0.9))
                if time.time() - seq_end_time > 1:
                    seq_end_time = time.time()
                    seq_num += 1
            elif seq_num == 3:
                # 待機
                send_cmd_direction(complex(0.0, 0.0))
                if time.time() - seq_end_time > 1:
                    seq_end_time = time.time()
                    seq_num += 1
            elif seq_num == 4:
                # 旋回
                send_cmd_direction(complex(0.0, -0.9))
                if time.time() - seq_end_time > 2:
                    seq_end_time = time.time()
                    seq_num += 1
            elif seq_num == 5:
                # 待機
                send_cmd_direction(complex(0.0, 0.0))
                if time.time() - seq_end_time > 1:
                    seq_end_time = time.time()
                    seq_num += 1
            elif seq_num == 6:
                # 旋回
                send_cmd_direction(complex(0.0, 0.9))
                if time.time() - seq_end_time > 1:
                    seq_end_time = time.time()
                    seq_num += 1
            elif seq_num == 7:
                # 待機
                send_cmd_direction(complex(0.0, 0.0))
                if time.time() - seq_end_time > 1:
                    seq_end_time = time.time()
                    seq_num = 0
                    moon_detected = False
        else:
            if follow_start:
                if len(circles) > 0:
                    # マーカーが存在する場合
                    # ライントレース
                    cy_max = [0, 0]
                    for i, circle in enumerate(circles):
                        if circle[1] > cy_max[1]:
                            cy_max[1] = circle[1]
                            cy_max[0] = i
                    target_pose = circles[cy_max[0]]
                    error = img_gray.shape[1]/2 - target_pose[0]
                    turn = error * 0.004

                    if time.time() - 1 > Timeout:
                        send_cmd_direction(complex(0.4, turn))
                else:
                    # マーカーを見失ったとき
                    if time.time() - 1 > Timeout:
                        send_cmd_direction(complex(0.0, 0.8))


        print("[robot_pose] " + str(robot_pose[0]) + ", " + str(robot_pose[1]) + ", " + str(robot_pose[2]))
        if loop_count % 5 == 0:
            pose = [robot_pose[0], robot_pose[1], robot_pose[2]]
            pose_list.append(pose)

        # CSV書き込み
        if save_map:
            with open('map.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerows(pose_list)
                print("[info] map saved")
            save_map = False



