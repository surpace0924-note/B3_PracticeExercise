#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'duc_tin'

import math
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

#######################################################
# from matplotlib import pyplot as plt
# import numpy as np

RANGE_POINT = 0.1 # [m] これ以上既存の点と離れていたら新規としてマップに追加

class map_list:
    def __init__(self, type, pos, br_id, br_rad, flag):
        self.type = type  # moonかcircle
        self.id = id  # id
        self.pos = pos  # [x,y]
        self.br_id = br_id  # id_list []
        self.br_rad = br_rad  # rad_list []
        self.flag = flag  # 初回処理

    def add_map(self, pos, br_id, br_rad):  # 新規点をマップに追加する処理
        self.pos.append(pos)
        self.br_id.append(br_id)
        self.br_rad.append(br_rad)
        #self.id.append(len(self.id))

    def calc_dist(self, point:[]):  # 既知のマップの点いずれかとの最小距離を求める
        r_min = -1
        # print(self.pos)
        for n in range(len(self.pos)):
            a = np.array([point[0] - self.pos[n][0], point[1] - self.pos[n][1]])
            r = np.linalg.norm(a)
            if r_min == -1:
                r_min = r
            elif r < r_min:
                r_min = r
        return r_min

    def reg(self, pts:[[]]):    # マップに登録するかどうかの判定
        if self.flag == 1:  # 初回処理(マップが空の場合)
            self.flag = 0
            for n in range(len(pts)):
                if self.type == "moon":
                    br_vecs = find_vec(pts[n], map_circle)  # 現在いる枝をid 0とする
                    br_ids = []
                    for m in range(len(br_vecs)):
                        br_ids.append(m)
                    self.add_map([pts[n][0], pts[n][1]], br_ids, br_vecs)
                else: # circle
                    self.add_map([pts[n][0], pts[n][1]], 0, 0)

        else: # 2回目以降
            for n in range(len(pts)):
                r = self.calc_dist(pts[n])  # 既知点いずれかとの最短距離

                if r > RANGE_POINT: # 新規点
                    if self.type == "moon":
                        #書き直し
                        br_ids = []
                        br_vecs = find_vec(pts[n], map_circle)
                        x = pts[n][0] - robot_pose[0]
                        y = pts[n][1] - robot_pose[1]
                        rad = math.atan2(y, x)  # 発見した月と現在のマシンがなす角度(ワールド座標基準)
                        minimum = 100
                        for m in range(len(br_vecs)): # 発見した枝のうち、角度が一番近いものは今いる枝とみなす
                            tmp = math.fabs(br_vecs[m] - rad)
                            num = 0
                            if tmp < minimum:
                                minimum = tmp
                                num = m
                        del br_vecs[num] # 現在いる枝を発見した枝リストから削除
                        tail = branch_list[len(branch_list)-1] # 末尾要素の取得
                        for m in range(0, len(br_vecs)-len(self.br_id), 1): # 0から未知個数分だけ
                            branch_list.append(tail + m)
                            br_ids.append(tail + m)
                        self.add_map([pts[n][0], pts[n][1]], br_ids, br_vecs)
                    else: # サークルの場合、枝の概念なし
                        self.add_map([pts[n][0], pts[n][1]], 0, 0)

                else: # 既知点
                    if self.type == "moon":
                        # 枝の追加確認 もし数が多ければ...
                        br_vecs = find_vec(pts[n], map_circle)
                        if len(br_vecs) > len(self.br_id):
                            tail = branch_list[len(branch_list) - 1] # 末尾idの取得
                            for m in range(0, len(br_vecs) - len(self.br_id), 1):  # 未知個数分だけ
                                branch_list.append(tail + m)







map_moon = map_list("moon", [], [], [], 1)
map_circle = map_list("circle", [], [], [], 1)
branch_list = [0]
branch_current = 0
# robot_pose = [0,0,0,0]  # x, y, theta, distance

def get_map(moon, cir):
    if len(cir) != 0:
        map_list.reg(map_circle, cir)

    if len(moon) != 0:
        map_list.reg(map_moon, moon)

    # x = [0,0]
    # y = [0,0]
    # x.clear()
    # y.clear()
    # for n in range(len(map_circle.pos)):
    #     x.append(map_circle.pos[n][0])
    #     y.append(map_circle.pos[n][1])
    # plt.scatter(x, y)
    # plt.show()

    return

'''
def toWorldCoordinate(machine_x, machine_y, yaw, point_cam): # マシンのX, Y, yaw, カメラで見た点群
    # カメラのマーカー座標を、マシン位置基準座標からワールド座標に直して返す
    # point_cam  は [[x,y], [x,y], ...] でおねがい
    point_w = []
    yaw = -yaw
    for n in point_cam:
        x = point_cam[n][0]
        y = point_cam[n][1]
        u = math.cos(yaw) * x + math.sin(yaw) * y
        v = -math.sin(yaw) * x + math.cos(yaw) * y
        point_w[n][0] = machine_x + u
        point_w[n][1] = machine_y + v
    return point_w
'''




def find_vec(moon, cir_l:map_list):
    # moon = [x,y]
    # cir_l = map_list型のcircle_data
    cir_near = []
    cir_near = cir_l.pos.copy()

    for n in range(len(cir_near)):
        x = cir_l.pos[n][0] - moon[0]  # X
        y = cir_l.pos[n][1] - moon[1]  # Y

        r = math.sqrt(pow(x, 2) + pow(y, 2))
        th = math.atan2(y, x)
        cir_near[n].append(r)
        cir_near[n].append(th)
    # これでcir_nearは、ワールド座標のxyと、あるmoonを基準にしたr,theta[x,y,r,theta]

    # 近い順に最大6個残す
    cir_near = sorted(cir_near, key=lambda x: x[2])
    if len(cir_near) > 6 :
        del cir_near[6:len(cir_l.pos)]
    #print ("near")
    #print (cir_near)

    # x = []
    # y = []
    # for n in range(len(cir_near)):
    #     x.append(cir_near[n][0])
    #     y.append(cir_near[n][1])
    # plt.scatter(x, y)
    # plt.show()

    # 角度順
    cir_near = sorted(cir_near, key=lambda x: x[3])
    # print ("rad")
    # print (cir_near)

    cir_br = [cir_near[0][3]]
    temp = cir_near[0][3]
    for n in range(len(cir_near)-1):
        # 角度差が20deg以下のとき、同じ方向と見なす
        if math.fabs(temp - cir_near[n+1][3]) > math.radians(20):
            cir_br.append(cir_near[n+1][3])
            temp = cir_near[n+1][3]
    return cir_br  # 月を基準とし、その近傍にある円の極座標theta[rad]のリスト[float, float, ...]を返す


'''

robot_pose_past = [0, 0, 0, 0]
circle_past = [[0,0]]
def pos_correct(circle):
    def calc_dist_xy(now:[], past_l:[[]]):  # 既知の点群の中で、もっとも近いものとの移動方向[x,y]を返す
        r_min = -1
        memo = -1
        for n in range(len(past_l)):
            a = np.array([now[0] - past_l[n][0], now[1] - past_l[n][1]])
            r = np.linalg.norm(a)
            if r_min == -1:
                r_min = r
                memo = n
            elif r < r_min:
                r_min = r
                memo = n
        # past_l[memo]が引数nowに最も近い[x,y]
        return [ now[0] - past_l[memo][0], now[1] - past_l[memo][1]]

    # カメラで検出した補正済みcircleのリスト[[x,y], ...]
    mov_machine = np.array(robot_pose) - np.array(robot_pose_past)
    mov_circle = []
    mov_cir_ave = np.array([0, 0])

    for n in range(len(circle)):  # circleの移動[x, y]がn本計算される
        a = np.array(calc_dist_xy(circle[n], circle_past))
        if np.linalg.norm(a) < 0.02:  # 0.02m 以上離れていたら新規円として無視
            mov_circle[n] = a

    for n in range(len(circle)):  # ↑それらの移動平均
        mov_cir_ave += mov_circle[n]
    mov_cir_ave /= len(mov_circle)    # circleの移動平均ベクトル

    ref = mov_cir_ave - mov_machine  # 補正値を計算し現在座標を補正(サークル移動量が正しいとする)
    robot_pose[0] += ref[0]
    robot_pose[1] += ref[1]
    robot_pose_past = robot_pose
    return "旋回時に対応できないのでゴミでした　使わないでください"
'''









'''
print(map.pos)
map_list.add_map(map, [3,3], [5], [7,7,7])
print(map.pos)
print(map.br_rad)

bbb = [55,55]
map_list.cir_reg(map, bbb)
print(map.pos)
'''


def find_vec2(moon, circles):
    # moon = [x,y]
    # circles = [[x,y], ... ]
    cir_near = []
    cir_near = circles.copy()
    for n in range(len(cir_near)):
        x = circles[n][0] - moon[0]  # X
        y = circles[n][1] - moon[1]  # Y
        r = math.sqrt(pow(x, 2) + pow(y, 2))
        th = math.atan2(y, x)
        cir_near[n].append(r)
        cir_near[n].append(th)
    # これでcir_nearは、ワールド座標のxyと、あるmoonを基準にしたr,theta[x,y,r,theta]
    # 近い順に最大6個残す
    cir_near = sorted(cir_near, key=lambda x: x[2])
    if len(cir_near) > 6 :
        del cir_near[6:len(circles)]
    # print ("near")
    # print (cir_near)
    # x = []
    # y = []
    # for n in range(len(cir_near)):
    #     x.append(cir_near[n][0])
    #     y.append(cir_near[n][1])
    # plt.scatter(x, y)
    # plt.show()
    # 角度順
    cir_near = sorted(cir_near, key=lambda x: x[3])
    # print ("rad")
    # print (cir_near)
    cir_br = [cir_near[0][3]]
    temp = cir_near[0][3]
    for n in range(len(cir_near)-1):
        # 角度差が20deg以下のとき、同じ方向と見なす
        if math.fabs(temp - cir_near[n+1][3]) > math.radians(20):
            cir_br.append(cir_near[n+1][3])
            temp = cir_near[n+1][3]
    return cir_br  # 月を基準とし、その近傍にある円の極座標theta[rad]のリスト[float, float, ...]を返す

###########################################################


# 角度を-piから+piに正規化する
def normalizeAngle(angle):
    result = math.fmod(angle + math.pi, 2.0 * math.pi)
    if result <= 0.0:
        return result + math.pi
    return result - math.pi


robot_pose = [0.0, 0.0, 0.0, 0.0]    # ロボットの自己位置[x, y, theta, distance]
robot_pose_history = []
moon_relative_pose = [0.0, 0.0]     # ロボット座標系での月の位置
moon_pose = [0.0, 0.0]              # ワールド座標系での月の位置

def pxXToMeterY(px_x, width):
    m_per_px = 0.000483
    tmp_x = m_per_px * px_x
    return -(tmp_x - width/2 * m_per_px)
def pxYToMeterX(px_y, height):
    m_per_px = 0.0005
    robot_to_screan = 0.13
    tmp_y = m_per_px * px_y
    return height * m_per_px - tmp_y + robot_to_screan

def toWorldCoordinate(robot_pose, point_cam):
    # マシンからの相対座標をワールド座標へ変換
    # point_cam  が [[x,y], [x,y], ...] ならば
    point_w = [0, 0]
    x = point_cam[0]
    y = point_cam[1]
    # u = np.cos(-robot_pose[2] + math.radians(90)) * x + np.sin(-robot_pose[2] + math.radians(90)) * y
    # v = -np.sin(-robot_pose[2] + math.radians(90)) * x + np.cos(-robot_pose[2] + math.radians(90)) * y
    u = np.cos(-(robot_pose[2] + math.radians(0))) * x + np.sin(-(robot_pose[2] + math.radians(0))) * y
    v = -np.sin(-(robot_pose[2] + math.radians(0))) * x + np.cos(-(robot_pose[2] + math.radians(0))) * y
    point_w[0] = -robot_pose[0] + u
    point_w[1] = -robot_pose[1] + v
    return point_w # ワールド座標でみたカメラのマーカー座標

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
    try:
        robot_pose[0] = float(response_list[0])
        robot_pose[1] = float(response_list[1])
        robot_pose[2] = normalizeAngle(float(response_list[2]))
        robot_pose[3] = float(response_list[3])
        # print("[cmd_wheel_speed] L:" + str(cmd_wheel_speed[0]) + ", R:" + str(cmd_wheel_speed[1]))
    except:
        print("[error]")



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
            # print(pose_list)

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
seq_end_dist = 0.0

loop_count = 0
pose_list = []
marker_list = []
next_save_distance = 0

intersection_pose_list = []

select_direction = 1

target_moon_pose = []
target_rad = 0.0
target_dist = 0.0

moon_detected_num = 0
turn_val = 0

branch_angle = []

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
        elif key == 49:     # 1: 進む方向
            select_direction = 0
        elif key == 50:     # 2: 進む方向
            select_direction = 1
        elif key == 51:     # 3: 進む方向
            select_direction = 2

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
        img_projection = cv2.warpPerspective(img_gray, M, (p_trans[3][0], p_trans[3][1]), borderValue=(130, 130, 130))
        cv2.imshow('img_projection', img_projection)

        # エッヂの検出
        img_edge = cv2.Canny(img_projection,100,200)
        # cv2.imshow('img_edge', img_edge)

        # # 2値化
        _, img_binarized = cv2.threshold(img_projection, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # cv2.imshow('img_binarized', img_binarized)

        # マーカー検出
        img_marker = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR) # 表示用画像

        # 画像の高さと幅
        fig_height, fig_width = img_projection.shape[0], img_projection.shape[1]

        # 月の検出
        moon_px = [0.0, 0.0, 0.0]    # [x, y, radius] [px]
        moon_pose = []  # [x, y] [m]
        cv_moons = cv2.HoughCircles(img_edge, cv2.HOUGH_GRADIENT, 1, 1000, param1=10, param2=15, minRadius=30, maxRadius=50)
        if cv_moons is not None:
            moon_px[0] = cv_moons[0][0][0]
            moon_px[1] = cv_moons[0][0][1]
            moon_px[2] = cv_moons[0][0][2]
            moon_relative_pose = [pxYToMeterX(moon_px[1], fig_height), pxXToMeterY(moon_px[0], fig_width)]
            moon_pose = toWorldCoordinate(robot_pose, moon_relative_pose)
            # 描画用画像の作成
            cv2.circle(img_marker, (moon_px[0], moon_px[1]), np.uint16(moon_px[2]), (0,255,255), 2) # 囲み線
            cv2.circle(img_marker, (moon_px[0], moon_px[1]), 2, (0,0,255), 3)                       # 中心点

        # 円の検出
        circle_px = []      # [[x, y, radius], []] [px]
        circle_pose = []    # [[x, y], []] [m]
        cv_circles = cv2.HoughCircles(img_edge, cv2.HOUGH_GRADIENT, 1, 40, param1=10, param2=12, minRadius=9, maxRadius=15)
        if cv_circles is not None:
            for cv_cir in cv_circles[0,:]:
                # 月内部の点はリストに追加しない
                if len(moon_px) != 0:
                    distance_to_moon = np.sqrt((moon_px[0] - cv_cir[0])**2 + (moon_px[1] - cv_cir[1])**2)
                    if distance_to_moon < moon_px[2]:
                        continue

                circle_px.append([cv_cir[0], cv_cir[1], cv_cir[2]])
                circle_pose.append(toWorldCoordinate(robot_pose, [pxYToMeterX(cv_cir[1], fig_height), pxXToMeterY(cv_cir[0], fig_width)]))
                # 描画用画像の作成
                cv2.circle(img_marker, (cv_cir[0],cv_cir[1]), np.uint16(cv_cir[2]), (255,255,0), 2) # 囲み線
                cv2.circle(img_marker, (cv_cir[0],cv_cir[1]), 2, (0,0,255), 3)                      # 中心点

        # マーカー検出の結果を表示
        cv2.imshow('img_marker', img_marker)

        # 月が見つかった
        if moon_detected == False:
            if len(moon_pose) != 0:
                moon_detected_num += 1
                if moon_detected_num > 3:
                    moon_detected = True

        # 走行制御
        if moon_detected:
            # 月を発見
            if seq_num == 0:
                # 停止
                send_cmd_direction(complex(0.0, 0.0))
                follow_start = False
                seq_end_time = time.time()
                seq_num += 1

            elif seq_num == 1:
                # 2秒経過するまで待機
                send_cmd_direction(complex(0.0, 0.0))
                if time.time() - seq_end_time > 2:
                    target_moon_pose = moon_pose
                    get_map([moon_pose], circle_pose)
                    branch_angle = find_vec2(moon_pose, circle_pose)

                    msg = "angles "
                    for i, b in enumerate(branch_angle):
                        msg += str(math.degrees(b)) + " "
                    print(msg)

                    # dist_to_moon = np.sqrt(moon_relative_pose[0]**2 + moon_relative_pose[1]**2)
                    dist_to_moon = moon_relative_pose[0]
                    target_dist = robot_pose[3] + dist_to_moon
                    turn_val = moon_relative_pose[1] * 4
                    seq_end_time = time.time()
                    seq_end_dist = robot_pose[3]
                    seq_num += 1

            elif seq_num == 2:
                # 月の位置まで前進
                send_cmd_direction(complex(0.45, turn_val))

                if robot_pose[3] > target_dist:
                    # 月に到着
                    print("[info] moon reach")
                    send_cmd_direction(complex(0.0, 0.0))
                    seq_end_time = time.time()
                    seq_num += 1

            elif seq_num == 3:
                # 進行方向へ旋回
                send_cmd_direction(complex(0.0, 0.8))

                msg = "angles "
                for i, b in enumerate(branch_angle):
                    msg += str(math.degrees(b)) + " "
                print(msg)
                print(math.degrees(robot_pose[2]))
                print(math.degrees(branch_angle[select_direction] - robot_pose[2]))
                print(select_direction)
                print(" ")

                if abs(robot_pose[2] - branch_angle[select_direction]) < math.radians(20):
                    print("[info] direction OK")
                    send_cmd_direction(complex(0.0, 0.0))
                    follow_start = True
                    seq_end_time = time.time()
                    seq_num = 0
                    moon_detected_num = 0
                    moon_detected = False

        follow_target = []
        if follow_start:
            # 点追従処理
            if len(circle_px) > 0:
                # マーカーが存在する場合
                # yが一番真ん中に近い点を算出
                tmp_cy = [0, 10000]
                for i, cir_px in enumerate(circle_px):
                    error_from_target = abs(cir_px[1] - fig_height/2)
                    if error_from_target < tmp_cy[1]:
                        tmp_cy[1] = error_from_target
                        tmp_cy[0] = i
                # 見つけた点と画像中心とのx成分の偏差を計算
                follow_target = toWorldCoordinate(robot_pose, circle_px[tmp_cy[0]])
                error = fig_width/2 - circle_px[tmp_cy[0]][0]
                send_cmd_direction(complex(0.45, error * 0.004))
            else:
                # マーカーを見失ったとき
                send_cmd_direction(complex(0.0, 0.8))

        if robot_pose[3] > next_save_distance:
            pose = [robot_pose[0], robot_pose[1], robot_pose[2]]
            robot_pose_history.append(pose)
            # robot_pose_history.append(follow_target)
            next_save_distance = robot_pose[3] + 0.05

        # CSV書き込み
        if save_map:
            # print(map_moon.pos)
            with open('map.csv', 'w') as f:
                writer = csv.writer(f)
                # writer.writerows(map_moon.pos)
                writer.writerows(robot_pose_history)
                print("[info] map saved")
            save_map = False
