# Chebyshev Linkage Shimulation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.widgets import Button

# 初期値、固定値
#  リンクの長さの初期値
oa_len_init = 1
ob_len_init = 2
ac_len_init = 2.5
bc_len_init = 2.5
cd_len_init = 2.5
#  スライダーの最小、最大値
oa_len_min = 0.5
ob_len_min = 1.5
bc_len_min = 1.5
ac_len_min = 1.5
cd_len_min = 1.5
oa_len_max = 2.0
ob_len_max = 3.0
bc_len_max = 3.0
ac_len_max = 3.0
cd_len_max = 5.0
#  表示色
line_color = "navy"
debug_color = "darkgrey"
point_color = "lightyellow"
text_color = "orangered"
track_color = "limegreen"
#  ほか、いろいろ
text_pos_offset = 0.1 # 点の名前を示すテキストの位置のオフセット
debag_option = False # デバッグ表示のオプション。表示したいときはTrueに変える。

# 変数
angle_o = 0.0  # ジョイントOの角度（初期値） [rad]　基準はx軸方向を0として半時計回りが正
angle_b = 0.0  # ジョイントBの角度（初期値） [rad]　基準はx軸方向を0として半時計回りが正
angle_a = 0.0  # リンクADの傾き角度 (初期値) [rad]　基準はx軸方向を0として半時計回りが正

len_oa = oa_len_init  #リンクOAの長さ
len_ob = ob_len_init  # OとBの間隔
len_ac = ac_len_init  # リンクACの長さ
len_bc = bc_len_init  # リンクBCの長さ
len_cd = cd_len_init  # リンクCDの長さ
point_o = np.array([0.0, 0.0])  # 点Oの座標。原点に固定。
point_a = np.array([0.0, 0.0])  # 点Aの座標（仮）
point_b = np.array([len_ob, 0.0])  #点Bの座標（仮）
point_c1 = np.array([0.0, 0.0])  #点Cの座標（仮）
point_c2 = np.array([0.0, 0.0])  #点Cの座標（仮）
point_c = np.array([0.0, 0.0])  #点Cの座標（仮）
point_d = np.array([0.0, 0.0])  #点Dの座標（仮）
track_D_x = None # 点Dの軌跡
track_D_y = None # 点Dの軌跡
data_valid_flag = True  # リンクの長さがおかしくて解なしのときFalse

# デバッグ
if debag_option:
    point_w_0 = np.array([0.0, 0.0])  #点Wの座標（仮） 交点の直線をデバッグ描画するための座標
    point_w_1 = np.array([0.0, 0.0])  #点Wの座標（仮） 交点の直線をデバッグ描画するための座標
    point_h = np.array([0.0, 0.0])  #点Wの座標（仮） 交点の直線をデバッグ描画するための座標


fig, ax = plt.subplots(figsize=(12, 12))

# デバッグ
if debag_option:
    # 点Cの候補円 AC側
    circle_c_a = plt.Circle((point_a[0], point_a[1]), len_ac, fill=False, color=debug_color)
    ax.add_artist(circle_c_a)
    # 点Cの候補円 BC側
    circle_c_b = plt.Circle((point_b[0], point_b[1]), len_bc, fill=False, color=debug_color)
    ax.add_artist(circle_c_b)
    # 点H (直線上)
    ax_point_h, = ax.plot(point_h[0], point_h[1], marker='o', color=debug_color)

# 円周上を進む点の座標を計算する関数
def generate_point(theta, r):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.array([x, y])

# 点Aの軌跡。円を描画する
circle_a = plt.Circle((point_o[0], point_o[1]), oa_len_init, fill=False, color=debug_color)
ax.add_artist(circle_a)

# デバッグ 交点の線
if debag_option:
    debug_line, =  ax.plot([point_w_0[0], point_w_1[0]], [point_w_0[1], point_w_1[1]], color=debug_color)

# 点Dの軌跡
ax_track_d, = ax.plot([0, 0], [0, 0], linewidth=5, color=track_color) # D

# 線 初期化用
ax_line_oa, = ax.plot([point_o[0], point_a[0]], [point_o[1], point_a[1]], linewidth=5, color=line_color) # OA
ax_line_ac, = ax.plot([point_a[0], point_c[0]], [point_a[1], point_c[1]], linewidth=5, color=line_color) # AC
ax_line_bc, = ax.plot([point_b[0], point_c[0]], [point_b[1], point_c[1]], linewidth=5, color=line_color) # BC
ax_line_cd, = ax.plot([point_c[0], point_d[0]], [point_c[1], point_d[1]], linewidth=5, color=line_color) # CD

# 初期化用の点を描画する
# 点A
point_a = generate_point(0, oa_len_init)
ax_point_a, = ax.plot(point_a[0], point_a[1], marker='o', color=point_color)
ax_text_a = ax.text( point_a[0]+text_pos_offset, point_a[1]+text_pos_offset, "A", color=text_color)  # 点の横にテキストを描画する
# 点O
ax_point_o, = ax.plot(point_o[0], point_o[1], marker='o', color=point_color)
ax_text_o = ax.text( point_o[0]+text_pos_offset, point_o[1]+text_pos_offset, "O", color=text_color)  # 点の横にテキストを描画する
# 点B
ax_point_b, = ax.plot(point_b[0], point_b[1], marker='o', color=point_color)
ax_text_b = ax.text( point_b[0]+text_pos_offset, point_b[1]+text_pos_offset, "B", color=text_color)  # 点の横にテキストを描画する
# 点C
ax_point_c, = ax.plot(point_c[0], point_c[1], marker='o', color=point_color)
ax_text_c = ax.text( point_c[0]+text_pos_offset, point_c[1]+text_pos_offset, "C", color=text_color)  # 点の横にテキストを描画する
# 点C
ax_point_d, = ax.plot(point_d[0], point_d[1], marker='o', color=point_color)
ax_text_d = ax.text( point_d[0]+text_pos_offset, point_d[1]+text_pos_offset, "D", color=text_color)  # 点の横にテキストを描画する

# Dの統計
ax_text_d_x_min_max = ax.text( 4, 1, "D.x  min " + "max " , color=text_color)
ax_text_d_y_min_max = ax.text( 4, 1-0.4, "D.y  min " + "max " , color=text_color)
ax_text_d_xy_ratio = ax.text( 4, 1-0.4*2, "D xy ratio" , color=text_color)

# 点Cの位置を求める。
def calc_point_c(len_ac, len_bc):
    # AC、BCが交わる条件
    if np.sqrt((point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2) >= len_ac + len_bc:
        return False

    # AC、BCを半径とする2つの円の交点は2つ存在する。この2点を結ぶ直線の係数を求める。
    a = 2 * (point_a[0] - point_b[0])  # 2(x0 - x1)
    b = 2 * (point_a[1] - point_b[1])  # 2(y0 - y1)
    c = len_ac**2 - len_bc**2 - (point_a[0]**2 + point_a[1]**2 - point_b[0]**2 - point_b[1]**2)  # r0^2 - r1^2 - (x0^2 + y0^2 - x1^2 -y1^2)
    d = np.abs(a * point_a[0] + b * point_a[1] + c)

    # デバッグ
    if debag_option:
        point_w_0[0] = -100
        point_w_0[1] = - a / b * point_w_0[0] - c / b
        point_w_1[0] = 100
        point_w_1[1] = - a / b * point_w_1[0] - c / b
        point_h[0] = - (a * d) / (a**2 + b**2) + point_a[0]
        point_h[1] = - (b * d) / (a**2 + b**2) + point_a[1]


    # 円と直線の交点を求める
    # 参考　https://qiita.com/tydesign/items/36b42465b3f5086bd0c5
    if (a**2 + b**2) * len_ac**2 - d**2 < 0:
        print("Something wrong with linkage length...")
        return False
    point_c1[0] = -(a * d - b * np.sqrt((a**2 + b**2) * len_ac**2 - d**2)) / (a**2 + b**2) + point_a[0]  # (a d - b SQRT((a^2 + b^2)r0^2 - D^2)) / (a^2 + b^2) + x0
    point_c1[1] = -(b * d + a * np.sqrt((a**2 + b**2) * len_ac**2 - d**2)) / (a**2 + b**2) + point_a[1]  # (b d + a SQRT((a^2 + b^2)r0^2 - D^2)) / (a^2 + b^2) + y0
    point_c2[0] = -(a * d + b * np.sqrt((a**2 + b**2) * len_ac**2 - d**2)) / (a**2 + b**2) + point_a[0]  # (a d + b SQRT((a^2 + b^2)r0^2 - D^2)) / (a^2 + b^2) + x0
    point_c2[1] = -(b * d - a * np.sqrt((a**2 + b**2) * len_ac**2 - d**2)) / (a**2 + b**2) + point_a[1]  # (b d - a SQRT((a^2 + b^2)r0^2 - D^2)) / (a^2 + b^2) + y0

    # update point_c
    if point_c1[1] > point_c2[1]:
        point_c[0] = point_c1[0]
        point_c[1] = point_c1[1]
    else:
        point_c[0] = point_c2[0]
        point_c[1] = point_c2[1]

    if point_c[0] > 10000 or point_c[1] > 10000:
        print("Point C position is extremely large...")
        return False

    return True

def length_between_2points(point1, point2):  # 入力はx,y座標
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calc_point_d(len_cd):
    vector_ac = np.array([point_c[0] - point_a[0], point_c[1] - point_a[1]])
    vector_cd = vector_ac * len_cd / length_between_2points(point_a, point_c)
    point_d[0] = point_c[0] + vector_cd[0]
    point_d[1] = point_c[1] + vector_cd[1]
    return True

# アニメーションを更新する関数
def update(frame, len_oa, len_ob, len_bc, len_ac, len_cd):
    angle_o = np.deg2rad(frame)

    # 点A
    tmp_result = generate_point(angle_o, len_oa)
    point_a[0] = tmp_result[0]
    point_a[1] = tmp_result[1]
    ax_point_a.set_data(point_a[0], point_a[1])
    ax_text_a.set_x(point_a[0]+text_pos_offset)
    ax_text_a.set_y(point_a[1]+text_pos_offset)

    # 線
    ax_line_oa.set_data([point_o[0], point_a[0]], [point_o[1], point_a[1]]) # OA
    ax_line_ac.set_data([point_a[0], point_c[0]], [point_a[1], point_c[1]]) # AC
    ax_line_bc.set_data([point_b[0], point_c[0]], [point_b[1], point_c[1]]) # BC
    ax_line_cd.set_data([point_c[0], point_d[0]], [point_c[1], point_d[1]]) # CD

    # 点Aの軌跡の円
    circle_a.set_radius(len_oa)

    # 点B
    point_b[0] = len_ob
    ax_point_b.set_data(point_b[0], point_b[1])
    ax_text_b.set_x(point_b[0]+text_pos_offset)
    ax_text_b.set_y(point_b[1]+text_pos_offset)

    # デバッグ
    if debag_option:
        circle_c_a.center = (point_a[0], point_a[1])  # 点Cの候補円 AC側
        circle_c_b.center = (point_b[0], point_b[1])  # 点Cの候補円 BC側
        ax_point_h.set_data(point_h[0], point_h[1])
        debug_line.set_data([point_w_0[0], point_w_1[0]], [point_w_0[1], point_w_1[1]]) # 交点の線

    # 点C
    data_valid_flag = calc_point_c(len_ac, len_bc)
    if  data_valid_flag is False:
        print("calc_point_c() is False")
        ax.set_facecolor("pink")
        return
    else:
        ax.set_facecolor("white")
    ax_point_c.set_data(point_c[0], point_c[1])
    ax_text_c.set_x(point_c[0]+text_pos_offset)
    ax_text_c.set_y(point_c[1]+text_pos_offset)

    # 点D
    calc_point_d(len_cd)
    ax_point_d.set_data(point_d[0], point_d[1])
    ax_text_d.set_x(point_d[0]+text_pos_offset)
    ax_text_d.set_y(point_d[1]+text_pos_offset)


    if frame < animation_len:
        global track_D_x
        global track_D_y
        if track_D_x is None:
            track_D_x = np.array([point_d[0]])
            track_D_y = np.array([point_d[1]])
        else:
            if track_D_x.shape[0] < animation_len:
                track_D_x = np.append(track_D_x, point_d[0])
                track_D_y = np.append(track_D_y, point_d[1])

    # 点Dの軌跡
    ax_track_d.set_data(track_D_x, track_D_y)

    # Dの軌跡の統計
    track_D_x_min = np.amin(track_D_x)
    track_D_x_max = np.amax(track_D_x)
    track_D_y_min = np.amin(track_D_y)
    track_D_y_max = np.amax(track_D_y)
    track_D_x_range = track_D_x_max - track_D_x_min
    track_D_y_range = track_D_y_max - track_D_y_min
    track_D_xy_ratio = track_D_y_range / track_D_x_range
    ax_text_d_x_min_max.set_text("D x min " + '{:.2f}'.format(track_D_x_min) + " max " + '{:.2f}'.format(track_D_x_max) + "  ... range " + '{:.2f}'.format(track_D_x_range))
    ax_text_d_y_min_max.set_text("D y min " + '{:.2f}'.format(track_D_y_min) + " max " + '{:.2f}'.format(track_D_y_max) + "  ... range " + '{:.2f}'.format(track_D_y_range))
    ax_text_d_xy_ratio.set_text("D y/x ratio " + '{:.2f}'.format(track_D_xy_ratio))

# グラフの設定
plt.title('Chebyshev Linkage Simulation')
plt.subplots_adjust(bottom=0.35)
ax.set_aspect('equal', adjustable='box') # アスペクト比を1:1にする
plt.axis([-5, 10, -5, 10])  # x軸、y軸の表示範囲
ax.set_xlabel('X') # x軸のラベルを設定する
ax.set_ylabel('Y') # y軸のラベルを設定する
#グリッド
ax.minorticks_on()     #補助線を追加
ax.grid( color="Grey")    #X軸とY軸のの主目盛線を色を設定して表示　         
ax.grid( axis="y", which= "minor",#Y軸の副目盛り線を表示
    linewidth="0.4",linestyle="--" ) #Y軸副目盛の線の太さ、線種を設定
ax.grid( axis="x", which= "minor",#Y軸の副目盛り線を表示
    linewidth="0.4",linestyle="--" ) #Y軸副目盛の線の太さ、線種を設定

# スライダーを作成する
ax_slider_oa = plt.axes([0.2, 0.25, 0.6, 0.03])
slider_oa = Slider(ax_slider_oa, 'OA length', oa_len_min, oa_len_max, valinit=oa_len_init)
ax_slider_ob = plt.axes([0.2, 0.20, 0.6, 0.03])
slider_ob = Slider(ax_slider_ob, 'OB length', ob_len_min, ob_len_max, valinit=ob_len_init)
ax_slider_bc = plt.axes([0.2, 0.15, 0.6, 0.03])
slider_bc = Slider(ax_slider_bc, 'BC length', bc_len_min, bc_len_max, valinit=bc_len_init)
ax_slider_ac = plt.axes([0.2, 0.10, 0.6, 0.03])
slider_ac = Slider(ax_slider_ac, 'AC length', ac_len_min, ac_len_max, valinit=ac_len_init)
ax_slider_cd = plt.axes([0.2, 0.05, 0.6, 0.03])
slider_cd = Slider(ax_slider_cd, 'CD length', cd_len_min, cd_len_max, valinit=cd_len_init)

# ボタンを作成する
ax_button_reset = plt.axes([0.02, 0.17, 0.08, 0.1]) # reset
btn_reset = Button(ax_button_reset, 'Reset', color="white", hovercolor='0.98')
ax_button_randomize = plt.axes([0.02, 0.05, 0.08, 0.1]) # randomize
btn_randomize = Button(ax_button_randomize, 'Ramdomize', color="white", hovercolor='0.98')

animation_len = 360
iterator_for_frames = np.linspace(0, animation_len, int(animation_len / 10))
animation_interval = 1

# スライダーの値が変更されたらアニメーションを更新する
def on_changed(val):
    len_oa = slider_oa.val
    len_ob = slider_ob.val
    len_bc = slider_bc.val
    len_ac = slider_ac.val
    len_cd = slider_cd.val
    global ani
    if ani is not None:
        ani.event_source.stop()

    # 点Dの軌跡の初期化
    global track_D_x
    global track_D_y
    track_D_x = None 
    track_D_y = None

    ani = FuncAnimation(fig, update, fargs=(len_oa, len_ob, len_bc, len_ac, len_cd), frames=iterator_for_frames, interval=animation_interval)
    plt.draw()

slider_oa.on_changed(on_changed)
slider_ob.on_changed(on_changed)
slider_bc.on_changed(on_changed)
slider_ac.on_changed(on_changed)
slider_cd.on_changed(on_changed)

def on_reset(event):
    slider_oa.set_val(oa_len_init)
    slider_ob.set_val(ob_len_init)
    slider_bc.set_val(bc_len_init)
    slider_ac.set_val(ac_len_init)
    slider_cd.set_val(cd_len_init)

def randomize_range(a, b):  # a以上b未満
    return (b - a) * np.random.rand() + a

def on_randomize(event):
    slider_oa.set_val(randomize_range(oa_len_min, oa_len_max))
    slider_ob.set_val(randomize_range(ob_len_min, ob_len_max))
    slider_bc.set_val(randomize_range(bc_len_min, bc_len_max))
    slider_ac.set_val(randomize_range(ac_len_min, ac_len_max))
    slider_cd.set_val(randomize_range(cd_len_min, cd_len_max))


btn_reset.on_clicked(on_reset)
btn_randomize.on_clicked(on_randomize)

# アニメーションを作成する
ani = FuncAnimation(fig, update, fargs=(oa_len_init, ob_len_init, bc_len_init, ac_len_init, cd_len_init), frames=iterator_for_frames, interval=animation_interval)

# アニメーションを表示する
plt.show()

# アニメーションを保存する場合
# ani.save('animation.gif', writer='pillow')


