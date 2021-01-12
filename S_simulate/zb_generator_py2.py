#!/usr/bin/env python 
# -*- coding: utf-8 -*-
__author__ = 'duc_tin'

from wsgiref.simple_server import make_server
from urllib.parse import parse_qs

"""
    A part of robot controller.
    Simulate the state after images are processed and positions or
    intersection are found. Because image processing is quite complicated,
    we keep it in parallel with decision maker (DFS).

    Both processes communicate with each other through web interface.
    Multi-processing and threading is also possible.
"""

course = [list(line.strip()) for line in
          """########################################
             #......................................#
             ##.#########.############.####.####.##.#
             ##....######.############.####..###.##.#
             #####..#####.############.#####.###.##.#
             ######.#####..............#####.###.##.#
             ######.#####.############.#####.###.##.#
             ###....#####.############...........##.#
             ###.########.############.#########.##.#
             #.....######.############.#########.##.#
             #.###.######.#.........................#
             #............#.############### #########
             ###.##########.############### #########
             ###.##########.......##................#
             ###.##########.########.####.#########.#
             #..............########.####.#########.#
             #.####.################.####.#########.#
             #.####.################.####.#########.#
             #....................##.#####.....####.#
             #.########.#########.##.#########.####.#
             #.########.#########.##.#########.####.#
             #.#####......#####.....................#
             #.......####.......#####################
             ########################################""".split('\n')]


class RoadManager:
    def __init__(self, course):
        self.course = course
        self.pos = (1, 1)
        self.search_vec = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def get_next(self, vec):
        if vec == (0, 0):
            self.pos = (1, 1)

        self.pos = x, y = self.pos[0] + vec[0], self.pos[1] + vec[1]
        canvec = [v for v in self.search_vec if self.course[x + v[0]][y + v[1]] == '.']
        return self.pos, canvec


# -----------------Network Application----------------------------------
def application(environ, start_response):
    # receive request from client
    query = parse_qs(environ['QUERY_STRING'])
    response_body = 'welcome'

    if 'direction' in query:
        request = [int(x) for x in query['direction'][0].split()]
        pos, vec = manager.get_next(request)
        pos = [str(x) for x in pos]
        vec = [str(x) for y in vec for x in y]
        response_body = ' '.join(pos + vec)

    status = '200 OK'
    response_headers = [('Content-Type', 'text/text'),
                        ('Content-Length', str(len(response_body)))]
    start_response(status, response_headers)

    return [response_body]


if __name__ == '__main__':
    # assign a manager for a specific course
    manager = RoadManager(course)

    # make us online forever
    make_server('127.0.0.1', 30002, application).serve_forever()

    print("server start")
