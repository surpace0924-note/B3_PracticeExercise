import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from statistics import mean, median, variance, stdev


# 配色リスト[黒，青，赤，緑，黄，紫，水色]
color_list = ["#000000", "#296fbc", "#cb360d",
              "#3d9435", "#e1aa13", "#a54675", "#138bae"]

# x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['xtick.direction'] = 'in'
# y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.0  # x軸主目盛り線の線幅
plt.rcParams['ytick.major.width'] = 1.0  # y軸主目盛り線の線幅
plt.rcParams['font.size'] = 9           # フォントの大きさ
plt.rcParams['axes.linewidth'] = 0.7    # 軸の線幅edge linewidth。囲みの太さ

fig = plt.figure(figsize=(23/2.54, 23/2.54))
ax = fig.add_subplot(1, 1, 1)

# データの読み込み
file_name = "map.csv"
p2 = np.genfromtxt(file_name, delimiter=',', filling_values=0)
x = p2[0:, 0]
y = p2[0:, 1]

ax.scatter(x, y, color=color_list[0])

# max_val = max(x)
# if max(y) > max_val:
#     max_val = max(y)

# min_val = min(x)
# if min(y) < min_val:
#     min_val = min(y)
# plt.xlim(min_val, max_val)
# plt.ylim(min_val, max_val)

ax.set_aspect('equal', adjustable='box')

plt.xlabel("x [m]", fontsize=10)
plt.ylabel("y [m]", fontsize=10)

plt.setp(ax.get_xticklabels(), fontsize=10)
plt.setp(ax.get_yticklabels(), fontsize=10)

ax.grid(ls="--")

# グラフタイトル
# plt.title('')

# グラフ範囲
# plt.xlim()
# plt.ylim(0.0, 0.7)

# 余白設定
# plt.subplots_adjust(left=0.105, right=0.98, bottom=0.21, top=0.95)

# グラフの凡例
# ax.legend(fancybox=False, framealpha=1, edgecolor="#000000",
#   loc = 'upper right', fontsize = 9)

# 表示
plt.show()