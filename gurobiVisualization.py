import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd


I = 7  # 车辆数
J = 16  # 最大任务次数（根据实际情况调整）
H = 480  # 规划时窗（分钟）
SWAP_T = 8
TRAVEL_AFTER_SWAP_T = 20


with open(f"results_I_{I}_J_{J}_H_{H}.pkl", "rb") as f:
    results = pickle.load(f)

# part 1: soc level by time
soc_level = []
for i in range(I):
    veh_soc = [0 for _ in range(H)]
    for j in range(J):
        if results['z'][i][j] > 0.5:
            end_t = int(results['T'][i][j])
            veh_soc[end_t - 30:end_t] = [int(results['E'][i][j])] * 30
            if results['x'][i][j] > 0.5:
                veh_soc[end_t:int(results['s'][i][j]+SWAP_T)] = (
                        [results['E'][i][j]] * (int(results['s'][i][j]+SWAP_T) - end_t))
                veh_soc[int(results['s'][i][j]+SWAP_T):int(results['s'][i][j]+SWAP_T+TRAVEL_AFTER_SWAP_T)] = (
                        [100] * (int(results['s'][i][j]+SWAP_T+TRAVEL_AFTER_SWAP_T) - int(results['s'][i][j]+SWAP_T)))
    soc_level.append(veh_soc)

plt.figure(figsize=(14, 6))
for i in range(I):
    plt.plot(soc_level[i], label=f"Vehicle {i+1}")
plt.xlabel("Time (minutes)")
plt.ylabel("SOC (%)")
plt.title("SOC Level by Time")
plt.legend()
plt.show()

# part 2: queue length by time
queue_length = [0 for _ in range(H)]
for i in range(I):
    for j in range(J):
        if results['x'][i][j] > 0.5:
            start_t = int(results['T'][i][j]) + 1
            end_t = int(results['s'][i][j]) + SWAP_T
            for t in range(start_t, end_t):
                if t < H:
                    queue_length[t] += 1

plt.figure(figsize=(14, 6))
plt.plot(queue_length, label="Queue Length")
plt.xlabel("Time (minutes)")
plt.ylabel("Queue Length")
plt.title("Queue Length by Time")
plt.yticks(range(0, max(queue_length) + 3, 1))
plt.legend()
plt.show()

# part 3: vehicle state by time
# 0: running, 1: waiting, 2: swapping, 3: traveling_to_station
vehicle_state = [[0 for _ in range(H)] for _ in range(I)]
for i in range(I):
    for j in range(J):
        if results['z'][i][j] > 0.5 and results['x'][i][j] > 0.5:
            mission_end_t = int(results['T'][i][j])
            swap_start_t = int(results['s'][i][j])
            swap_end_t = int(results['s'][i][j]) + SWAP_T
            for t in range(mission_end_t, swap_start_t):
                if t < H:
                    vehicle_state[i][t] = 1
            for t in range(swap_start_t, swap_end_t):
                if t < H:
                    vehicle_state[i][t] = 2
            for t in range(swap_end_t, swap_end_t + TRAVEL_AFTER_SWAP_T):
                if t < H:
                    vehicle_state[i][t] = 3

# plot it in the gantt chart
event_colors = {
    0: 'grey',  # working
    1: 'blue',  # waiting
    2: 'green',  # swapping
    3: 'yellow'  # traveling_to_station
}

fig, ax = plt.subplots()
gap = 0.2

# 遍历每个对象
for i, sequence in enumerate(vehicle_state):
    start_time = 0
    # 遍历序列中的每个事件
    for j, event in enumerate(sequence):
        # 创建矩形块表示事件
        rect = patches.Rectangle(
            (start_time, i + gap / 2),  # x坐标不变，y坐标增加间距
            1,  # 宽度
            1 - gap,  # 高度减少间距
            linewidth=0,  # 去掉默认的边框
            edgecolor='none',  # 去掉默认的边框
            facecolor=event_colors[event]
        )
        ax.add_patch(rect)

        # 在矩形的顶部和底部绘制水平线来模拟上下边框
        ax.plot(
            [start_time, start_time + 1],  # x范围
            [i + gap / 2, i + gap / 2],  # 下边框的y位置
            color='black',  # 边框颜色
            linewidth=1  # 边框宽度
        )
        ax.plot(
            [start_time, start_time + 1],  # x范围
            [i + gap / 2 + (1 - gap), i + gap / 2 + (1 - gap)],  # 上边框的y位置
            color='black',  # 边框颜色
            linewidth=1  # 边框宽度
        )

        start_time += 1

ax.set_xlim(0, H)
ax.set_ylim(0, I)
ax.set_xlabel('Time')
ax.set_ylabel('Vehicles')
ax.set_yticks([i+1 for i in range(I)])
# ax.set_yticks([i + 0.5 for i in range(I)])
# ax.set_yticklabels(['time' for i in range(I)])
ax.set_title('Gantt Chart of Events')
plt.tight_layout()

plt.show()

# part 4: check queue length and remaining soc at every decision point
# queue length
dec_q_length = []
dec_result = []
dec_soc = []
for i in range(I):
    dec_q_v_length = []
    dec_v_result = []
    dec_v_soc = []
    for j in range(J):
        if results['z'][i][j] > 0.5:
            ind = int(results['T'][i][j]) if int(results['T'][i][j]) < H else H - 1
            dec_q_v_length.append(queue_length[ind])
            dec_v_result.append(results['x'][i][j])
            dec_v_soc.append(results['E'][i][j])
    dec_q_length.append(dec_q_v_length)
    dec_result.append(dec_v_result)
    dec_soc.append(dec_v_soc)

check_dict = {
    'q_length': dec_q_length,
    'result': dec_result,
    'soc': dec_soc
}

whole_q_length = []
whole_result = []
whole_soc = []
for i in range(I):
    for j in range(len(check_dict['q_length'][i])):
        whole_q_length.append(check_dict['q_length'][i][j])
        whole_result.append(check_dict['result'][i][j])
        whole_soc.append(check_dict['soc'][i][j])
df = pd.DataFrame({
    'result': whole_result,
    'q_length': whole_q_length,
    'soc': whole_soc
})

Q_max = 5
T_d = 80
T_e = 35
alpha = 0.8

df['soc_score'] = (T_d - df['soc']) / (T_d - T_e)
df['q_score'] = (Q_max - df['q_length']) / Q_max
df['score'] = alpha * df['soc_score'] + (1-alpha) * df['q_score']

print(f'Objective: {30 * sum([sum(l) for l in results["z"]])}')
