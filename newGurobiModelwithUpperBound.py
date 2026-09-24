from gurobipy import Model, GRB, quicksum
import pickle

# 参数设置
I = 10
H = 540
T_run = 35
T_swap = 10
T_to_station = T_from_station = 10
C_init = C_swap = 80
Delta = 10
C_min = 25
J = H // T_run    # 最大任务数
print(J)
K = (C_swap - C_min) // Delta    # 最大连续执行任务数

# 计算单车最大任务数N及达到N趟的车辆数上界B
swapExtraTime = T_to_station + T_swap + T_from_station   # 换电额外占用
N = 0
for m in range(J):   # 枚举换电次数
    if H - m * swapExtraTime < 0:    # 此处不考虑换电等待时间
        break
    task_cap = min((m + 1) * K, (H - m * swapExtraTime) // T_run)
    N = max(N, task_cap)

if N <= K:          # 任务数不超过最大连续执行任务数，不需要换电，直接使用I个车辆
    B = I
else:
    minSwaps = (N + K - 1) // K - 1  # 最小换电次数   等价于  n // k 向上取整 再减1
    earliestService = T_run + T_to_station
    latestService = H - T_swap - T_from_station - T_run
    stationCapacity = max(0, (latestService - earliestService) // T_swap + 1)   # 换电站最多换电次数
    classBoundSum = 0   # 记录能够完成最大任务数量N的车辆数上界

    # 枚举完成N趟任务的可能换电次数m
    for m in range(minSwaps, N):
        if N > (m + 1) * K or N * T_run + m * swapExtraTime > H:
            continue

        waiting_slack = H - N * T_run - m * swapExtraTime
        window_capacity = waiting_slack // T_swap + 1
        class_bound = I

        # 每次换电时间窗能够容纳车辆数
        for r in range(1, m + 1):
            lower_position = max(r, N - (m - r + 1) * K)   # 第r次换电时，已完成的任务数下限
            upper_position = min(r * K, N - (m - r + 1))   # 第r次换电时，已完成的任务数上限
            position_count = max(0, upper_position - lower_position + 1)
            class_bound = min(class_bound, position_count * window_capacity)

        # 累加当前换电次数m的车辆数上界
        classBoundSum += class_bound

    B = min(I, stationCapacity // minSwaps, classBoundSum)

tripUpperBound = I * (N - 1) + B       # 车队最大总任务数
R = stationCapacity   # 最多换电服务次数
J = N   # 更新后的最大任务数
print(N)

# 时间变量上界和大M
T_upper = H + J * (T_run + T_swap)
M = T_upper + H + T_swap  + T_run

# 模型
model = Model("Battery_Swap_Scheduling")

# 创建变量
T = model.addVars(I, J, lb=0, ub=T_upper, vtype=GRB.CONTINUOUS, name="task_finish_time")   # i辆车在第j趟任务的完成时间
z = model.addVars(I, J, vtype=GRB.BINARY, name="finish_or_not")   # i辆车是否在规划视窗内完成第j趟任务
a = model.addVars(I, J - 1, R, vtype=GRB.BINARY, name="swap_or_not")    # 换电服务变量  a[i,j,r]=1 表示第i辆车在第j趟任务后占据了第r个换电服务
u = model.addVars(R, lb=0, ub=latestService, vtype=GRB.CONTINUOUS, name="swap_start_time")   # 第r个换电服务的开始时间

# 目标函数：最大化总运行时间
model.setObjective(quicksum(z[i, j] for i in range(I) for j in range(J)), GRB.MAXIMIZE)

# 1. 初始任务约束
for i in range(I):
    model.addConstr(T[i, 0] == T_run, name=f"initial_task_time_{i}")
    model.addConstr(z[i, 0] == 1, name=f"initial_task_{i}")

# 2. 任务连续性和时间窗约束
for i in range(I):
    for j in range(J):
        model.addConstr(T[i, j] <= H + M * (1 - z[i, j]), name=f"time_window_{i}_{j}")
        model.addConstr(T[i, j] >= (j + 1) * T_run, name=f"earliest_time_{i}_{j}")
        if j > 0:
            model.addConstr(z[i, j] <= z[i, j - 1], name=f"task_continuity_{i}_{j}")

# 3. 换电事件和任务时间更新约束
for i in range(I):
    for j in range(J - 1):
        x_ij = quicksum(a[i, j, r] for r in range(R))  # 是否换电

        # 只有下一趟执行时，才允许在当前趟后换电
        model.addConstr(x_ij <= z[i, j + 1], name=f"swap_activation_{i}_{j}")

        # 不换电时，下一趟完成时间等于当前完成时间加单趟时间
        model.addConstr(T[i, j + 1] >= T[i, j] + T_run, name=f"task_time_lb_{i}_{j}")
        model.addConstr(T[i, j + 1] <= T[i, j] + T_run + M * x_ij, name=f"task_time_ub_{i}_{j}")

        for r in range(R):
            # a=1时，换电服务不能早于车辆到达换电站
            model.addConstr(u[r] >= T[i, j] + T_to_station - M * (1 - a[i, j, r]), name=f"swap_release_{i}_{j}_{r}")

            # a=1时，下一趟完成时间由换电开始时间决定
            next_finish = u[r] + T_swap + T_from_station + T_run
            model.addConstr(T[i, j + 1] >= next_finish - M * (1 - a[i, j, r]), name=f"swap_time_lb_{i}_{j}_{r}")
            model.addConstr(T[i, j + 1] <= next_finish + M * (1 - a[i, j, r]), name=f"swap_time_ub_{i}_{j}_{r}")

# 4. 电池续航约束
# 每块满电电池最多连续执行K趟任务
for i in range(I):
    for j in range(K, J):
        recent_swaps = quicksum(a[i, k, r] for k in range(j - K, j) for r in range(R))
        model.addConstr(z[i, j] <= recent_swaps, name=f"battery_window_{i}_{j}")

# 5. 换电站容量约束
for r in range(R):
    slot_used = quicksum(a[i, j, r] for i in range(I) for j in range(J - 1))
    model.addConstr(slot_used <= 1, name=f"slot_capacity_{r}")
    model.addConstr(u[r] <= latestService * slot_used, name=f"unused_slot_{r}")
    model.addConstr(u[r] >= earliestService * slot_used, name=f"earliest_slot_{r}")

# 6. 换电服务位置顺序约束
for r in range(R - 1):
    current_used = quicksum(a[i, j, r] for i in range(I) for j in range(J - 1))
    next_used = quicksum(a[i, j, r + 1] for i in range(I) for j in range(J - 1))
    model.addConstr(next_used <= current_used, name=f"slot_continuity_{r}")
    model.addConstr(u[r + 1] >= u[r] + T_swap - M * (1 - next_used), name=f"slot_sequence_{r}")


# # 7. 车辆对称性约束
# for i in range(I - 1):
#     model.addConstr(quicksum(z[i, j] for j in range(J)) >= quicksum(z[i + 1, j] for j in range(J)), name=f"vehicle_symmetry_{i}")

# 7. 车辆对称性约束
for i in range(I - 1):
    for j in range(J):
        model.addConstr(z[i, j] >= z[i + 1, j], name=f"vehicle_symmetry_{i}_{j}")

# 8. 有效不等式
# 每辆车最多完成N趟
for i in range(I):
    model.addConstr(quicksum(z[i, j] for j in range(J)) <= N, name=f"single_vehicle_upper_bound_{i}")

# 最多B辆车能够完成N趟
if N > 0:
    model.addConstr(quicksum(z[i, N - 1] for i in range(I)) <= B, name="max_vehicles_at_N")

# 车队总任务数上界
model.addConstr(quicksum(z[i, j] for i in range(I) for j in range(J)) <= tripUpperBound, name="fleet_trip_upper_bound")

# 时间约束
for i in range(I):
    model.addConstr(
        T_run * quicksum(z[i, j] for j in range(J)) + swapExtraTime * quicksum(a[i, j, r] for j in range(J - 1) for r in range(R))
        <= H, name=f"vehicle_time_capacity_{i}"
    )

print(f"单车最大任务数N：{N}")
print(f"完成N趟的车辆数上界B：{B}")
print(f"车队总任务数上界：{tripUpperBound}")
print(f"目标函数上界：{tripUpperBound * T_run} 分钟")

# 求解设置
model.setParam("TimeLimit", 600)
model.setParam("MIPGap", 0.001)
model.optimize()

# 结果输出和保存
if model.SolCount > 0:
    print(f"当前最好总搬运次数：{model.ObjVal} 次")
    print(f"当前最优解：{model.ObjBound} 次")
    print(f"当前MIPGap：{model.MIPGap}")

    T_results = []
    z_results = []
    x_results = []
    s_results = []
    E_results = []

    for i in range(I):
        T_results.append([T[i, j].X for j in range(J)])
        z_results.append([int(round(z[i, j].X)) for j in range(J)])
        x_results.append([0 for j in range(J)])
        s_results.append([None for j in range(J)])

    station_schedule = []
    for i in range(I):
        for j in range(J - 1):
            for r in range(R):
                if a[i, j, r].X > 0.5:
                    x_results[i][j] = 1
                    s_results[i][j] = u[r].X
                    station_schedule.append([r, i, j, u[r].X, u[r].X + T_swap])
                    break

    station_schedule.sort()

    for i in range(I):
        energy = C_init
        vehicle_energy = []
        for j in range(J):
            if z_results[i][j] == 0:
                vehicle_energy.append(None)
            else:
                energy = energy - Delta
                vehicle_energy.append(energy)
                if j < J - 1 and x_results[i][j] == 1:
                    energy = C_swap
        E_results.append(vehicle_energy)

    results = {
        "T": T_results,
        "s": s_results,
        "x": x_results,
        "E": E_results,
        "z": z_results,
        "station_schedule": station_schedule,
        "objective": model.ObjVal
    }

    # with open(f"compact_results_I_{I}_J_{J}_H_{H}.pkl", "wb") as f:
    #     pickle.dump(results, f)

    print(f"结果已保存到 compact_results_I_{I}_J_{J}_H_{H}.pkl")
else:
    print("求解失败，未找到可行解")