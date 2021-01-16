#!/usr/bin/env python 
# -*- coding: utf-8 -*-
__author__ = 'duc_tin'

# Commands:
#	w-Move forward
#	a-Move left
#	d-Move right
#	s-Move back
#	x-Stop

from BrickPi import *  # import BrickPi.py file to use BrickPi operations
from wsgiref.simple_server import make_server
from urlparse import parse_qs


class HTTPdaemon:
    """Use this class to control real raspberry"""
    def __init__(self, brickpi_handler):
        self.babe = brickpi_handler
        self.cache = None

    def application(self, environ, start_response):
        # receive request from client
        query = parse_qs(environ['QUERY_STRING'])
        response_body = ' Anything you want to send back '

        if 'action' in query:
            action = query['action'][0]
            if action == 'forward':
                self.babe.fwd()
            elif action == 'left':
                self.babe.left()
            elif action == 'right':
                self.babe.right()
            elif action == 'backward':
                self.babe.back()
            elif action == 'stop':
                self.babe.stop()
            response_body = ' RPi has just received action '+action

            BrickPiUpdateValues()

        status = '200 OK'
        response_headers = [('Content-Type', 'text/html'),
                            ('Content-Length', str(len(response_body)))]
        start_response(status, response_headers)

        return [response_body]


class Babe:
    def __init__(self, speed, action_timeout):
        BrickPiSetup()                          # setup the serial port for communication
        self.motor1 = PORT_B
        self.motor2 = PORT_C
        BrickPi.MotorEnable[self.motor1] = 1    # Enable the Motor A
        BrickPi.MotorEnable[self.motor2] = 1    # Enable the Motor B
        BrickPiSetupSensors()                   # Send the properties of sensors to BrickPi
        self.speed = speed                      # Set the (speedA, speedB)
        self.set_timeout(action_timeout)        # If no new action received,
                                                # repeat the last action for action_timeout second

    def set_timeout(self, timeout):
        self.timeout = timeout
        BrickPi.Timeout = timeout

    def set_speed(self,(_left, _right)):
        self.speed = _left, _right

    # Move Forward
    def fwd(self):
        BrickPi.MotorSpeed[self.motor1] = self.speed[0]
        BrickPi.MotorSpeed[self.motor2] = self.speed[0]

    # Move Left
    def left(self):
        BrickPi.MotorSpeed[self.motor1] = self.speed[0]
        BrickPi.MotorSpeed[self.motor2] = -self.speed[1]

    # Move Right
    def right(self):
        BrickPi.MotorSpeed[self.motor1] = -self.speed[0]
        BrickPi.MotorSpeed[self.motor2] = self.speed[1]

    # Move backward
    def back(self):
        BrickPi.MotorSpeed[self.motor1] = -self.speed[0]
        BrickPi.MotorSpeed[self.motor2] = -self.speed[1]

    # Stop
    def stop(self):
        BrickPi.MotorSpeed[self.motor1] = 0
        BrickPi.MotorSpeed[self.motor2] = 0

if __name__ == '__main__':
    brick = Babe(speed=(50,50), action_timeout=5)
    daemon = HTTPdaemon(brickpi_handler=brick)
    host, port = '192.168.11.3', 8888
    print("RPi start serving http://%s:%s" % (host, port))
    make_server(host, port, daemon.application).serve_forever()
