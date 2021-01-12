#!/usr/bin/env python 
# -*- coding: utf-8 -*-

__author__ = 'duc_tin'

import numpy as np
import urllib, requests
import cv2
import random
import time


def send_cmd(action):
    address = 'http://%s:%s' % (ip, port)
    request = '/?action=%s' % action
    requests.post(address + request)


ip, port = '0.0.0.0', 8888

# stream = urllib.request.urlopen('http://%s:%s/?action=stream' % (ip, port))
stream = urllib.request.urlopen('http://0.0.0.0:8888/?action=stream')
bytes = ''
count = 0

pi_commands = ['forward', 'left', 'right', 'backward']
Timeout = 2     # second
t0 = 0

while 1:
    bytes += stream.read(1024)
    a = bytes.find('\xff\xd8')
    b = bytes.find('\xff\xd9')
    """take video from stream """
    if not (a != -1 and b != -1):
        continue

    # Simulation is kept as simple as possible, it's not a real
    # mjpeg stream but a single jpg image.
    # The below line only needed when receiving img from simulation
    #stream = urllib.request.urlopen('http://0.0.0.0:8888/?action=stream')

    jpg = bytes[a:b + 2]
    bytes = bytes[b + 2:]

    img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.CV_LOAD_IMAGE_COLOR)

    cv2.imshow('received', img)
    key = cv2.waitKey(80)

    if key == 27:   # break when press ESC
        break

    if time.time() - t0 > Timeout:
        cmd = random.choice(pi_commands)
        Timeout = random.randint(1, 5)
        t0 = time.time()

    send_cmd(cmd)
