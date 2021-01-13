#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'duc_tin'

import numpy as np
import urllib.request, urllib.error, urllib.parse, requests
import cv2
import random
import time


def send_cmd_direction(vdirect):
    """ 走行方向指令(複素数で指定) """
    address = 'http://%s:%s' % (ip, port)
    vdirect = '{} {}'.format(int(vdirect.real), int(vdirect.imag))
    request = '/?direction=%s' % vdirect
    # print(address + request)
    requests.post(address + request)


# ------ Main --------------

# HTTPサーバのポート設定　============
ip, port = '127.0.0.1', 8888

# 画像取得　============
stream = urllib.request.urlopen('http://%s:%s/?action=stream' % (ip, port))
bytes = bytes(b'')
count = 0

# 指令の指定　============
pi_commands = ['forward', 'left', 'right', 'backward']
pi_commands_vec = [complex(1, 0), complex(0, 1), complex(0, -1), complex(-1, 0)]
Timeout = 2  # second
t0 = 0

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
    stream = urllib.request.urlopen('http://%s:%s/?action=stream' % (ip, port))

    jpg = bytes[a:b + 2]
    bytes = bytes[b + 2:]

    img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
    # cv2.imshow('received', img)

    # ここに画像処理を追加
    #######################################
    # グレースケール化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('img_gray', img_gray)

    # 射影変換
    # 0 1
    # 2 3
    p_original = np.float32([[0, 0], [img_gray.shape[1], 0],
                             [0 - 850, img_gray.shape[0]], [img_gray.shape[1] + 850, img_gray.shape[0]]])
    p_trans = np.float32([[0, 0], [img_gray.shape[1], 0],
                          [0, img_gray.shape[0] + 130], [img_gray.shape[1], img_gray.shape[0] + 130]])
    M = cv2.getPerspectiveTransform(p_original, p_trans)
    img_projection = cv2.warpPerspective(img_gray, M, (p_trans[3][0], p_trans[3][1]), borderValue=(255, 255, 255))
    # cv2.imshow('img_projection', img_projection)

    # エッヂの検出
    img_edge = cv2.Canny(img_projection,10,100)
    # cv2.imshow('img_edge', img_edge)

    # マーカー検出
    img_marker = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR) # 表示用画像

    moons = []
    cv_moons = cv2.HoughCircles(img_edge, cv2.HOUGH_GRADIENT, 1, 1000, param1=10, param2=15, minRadius=30, maxRadius=50)
    if cv_moons is not None:
        for i in cv_moons[0,:]:
            moons.append([i[0], i[1], i[2]])
            cv2.circle(img_marker, (i[0],i[1]), np.uint16(i[2]), (0,255,255), 2)    # 囲み線
            cv2.circle(img_marker, (i[0],i[1]), 2, (0,0,255), 3)                    # 中心点
        print(moons)

    circles = []
    cv_circles = cv2.HoughCircles(img_edge, cv2.HOUGH_GRADIENT, 1, 40, param1=10, param2=10, minRadius=8, maxRadius=20)
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

        print(circles)


    cv2.imshow('img_marker', img_marker)



    #######################################
    key = cv2.waitKey(80)

    if key == 27:  # break when press ESC
        break

    if time.time() - t0 > Timeout:
        # 現在はランダムに走行指令を生成　<= 画像処理によって指令を決定するプログラミングを追加
        cmd_vdirect = complex(0, 0)

        Timeout = random.randint(1, 5)
        t0 = time.time()
        print("vdirect:", cmd_vdirect)  # 複素数の走行指令（方向）

    send_cmd_direction(cmd_vdirect)
