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
    cv2.imshow('received', img)
    # ここに画像処理を追加

    key = cv2.waitKey(80)

    if key == 27:  # break when press ESC
        break

    if time.time() - t0 > Timeout:
        # 現在はランダムに走行指令を生成　<= 画像処理によって指令を決定するプログラミングを追加
        cmd_vdirect = random.choice(pi_commands_vec)

        Timeout = random.randint(1, 5)
        t0 = time.time()
        print("vdirect:", cmd_vdirect)  # 複素数の走行指令（方向）

    send_cmd_direction(cmd_vdirect)
