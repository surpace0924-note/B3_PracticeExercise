import math
from matplotlib import pyplot as plt
import numpy as np

RANGE_POINT = 0.02 # [m] これ以上既存の点と離れていたら新規としてマップに追加

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
        print(self.pos)
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

    x = [0,0]
    y = [0,0]
    x.clear()
    y.clear()
    for n in range(len(map_circle.pos)):
        x.append(map_circle.pos[n][0])
        y.append(map_circle.pos[n][1])
    plt.scatter(x, y)
    plt.show()

    return

'''
def circle_calc(machine_x, machine_y, yaw, point_cam): # マシンのX, Y, yaw, カメラで見た点群
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

    x = []
    y = []
    for n in range(len(cir_near)):
        x.append(cir_near[n][0])
        y.append(cir_near[n][1])
    plt.scatter(x, y)
    plt.show()

    # 角度順
    cir_near = sorted(cir_near, key=lambda x: x[3])
    print ("rad")
    print (cir_near)

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