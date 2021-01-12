#!/usr/bin/env python 
# -*- coding: utf-8 -*-

__author__ = 'duc_tin'

import requests
import turtle as tt

"""
    This is DFS module. (dfsのためのモジュール)
    It works independently with robot controller.
    Robot controller handles images to get intersection and its position.
    This module only receive processed data and make decision:
        * where to go
        * when to stop
"""


class Comunicator:
    def __init__(self, ip, port):
        self.address = 'http://%s:%s' % (ip, port)
        self.intersections = []
        self.delta = 0.1

    def get_data(self, vdirect):
        vdirect = '{} {}'.format(int(vdirect.real), int(vdirect.imag))
        request = '/?direction=%s' % vdirect
        print(self.address + request)
        data = requests.get(self.address + request)

        pre = [int(x) for x in data.text.split()]
        raw = [complex(*pre[i:i + 2]) for i in range(0, len(pre), 2)]
        position, candidate = raw[0], raw[1:]
        return position, candidate

    def intersect_checker(self):
        """This job has been done in main"""
        pass


class Plotter:
    def __init__(self, w=800, h=600, start_x=100, start_y=100):
        self.wn = tt.Screen()
        tt.setup(width=w, height=h, startx=start_x, starty=start_y)
        self.me = tt.Turtle()

        # one of “arrow”, “turtle”, “circle”, “square”, “triangle”, “classic”
        self.me.shape('turtle')

        self.me.pensize(3)
        self.me.speed(1)
        self.speed_vec = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.last_spee = (0, 1)
        self.sign = -1

    def start_at(self, x, y):
        self.me.penup()
        self.me.setposition(x, y)
        self.me.pendown()

    def angle(self, v1, v2):
        ind1, ind2 = self.speed_vec.index(v1), self.speed_vec.index(v2)
        if ind1 - ind2 in (-1, 3):
            return 90
        if ind1 - ind2 in (1, -3):
            return -90
        if ind1 == ind2:
            return 0

        self.sign *= -1
        return 180 * self.sign

    def goto(self, v, dir):
        speed_vec = v.real, v.imag
        angle = self.angle(self.last_spee, speed_vec)
        self.last_spee = speed_vec
        self.me.left(angle)
        if dir == 1:
            self.me.pencolor('#000000')
        else:
            self.me.pencolor('#FF0000')
        self.me.forward(10)
