from gurobipy import Model, GRB, quicksum
import pickle

# 参数设置
I = 20
H = 720
T_run = 30
T_swap = 8
T_trip = 30
T_to_station = 10
T_from_station = 10
C_init = 100
C_swap = 100
Delta = 10
C_min = 25
J = H // T_run    # 最大任务数
K = (C_swap - C_min) // Delta    # 最大连续执行任务数

# 换电服务位置数量
earliest_swap = T_run + T_to_station
latest_swap = H - T_swap - T_from_station - T_run
R = (latest_swap - earliest_swap) // T_swap + 1   # 最多换电服务次数

# 时间变量上界和大M
T_upper = H + J * (T_run + T_swap + T_trip)
M = T_upper + H + T_swap + T_trip + T_run

# 创建模型
model = Model("Battery_Swap_Scheduling_Compact")

# 创建变量
T = model.addVars(I, J, lb=0, ub=T_upper, vtype=GRB.CONTINUOUS, name="T")
z = model.addVars(I, J, vtype=GRB.BINARY, name="z")
a = model.addVars(I, J - 1, R, vtype=GRB.BINARY, name="a")  # 换电服务变量
u = model.addVars(R, lb=0, ub=latest_swap, vtype=GRB.CONTINUOUS, name="u")

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
            model.addConstr(u[r] >= T[i, j] + T_to_station - M * (1 - a[i, j, r]),
                            name=f"swap_release_{i}_{j}_{r}")

            # a=1时，下一趟完成时间由换电开始时间决定
            next_finish = u[r] + T_swap + T_from_station + T_run
            model.addConstr(T[i, j + 1] >= next_finish - M * (1 - a[i, j, r]),
                            name=f"swap_time_lb_{i}_{j}_{r}")
            model.addConstr(T[i, j + 1] <= next_finish + M * (1 - a[i, j, r]),
                            name=f"swap_time_ub_{i}_{j}_{r}")

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
    model.addConstr(u[r] <= latest_swap * slot_used, name=f"unused_slot_{r}")
    model.addConstr(u[r] >= earliest_swap * slot_used, name=f"earliest_slot_{r}")

# 6. 换电服务位置顺序约束
for r in range(R - 1):
    current_used = quicksum(a[i, j, r] for i in range(I) for j in range(J - 1))
    next_used = quicksum(a[i, j, r + 1] for i in range(I) for j in range(J - 1))
    model.addConstr(next_used <= current_used, name=f"slot_continuity_{r}")
    model.addConstr(u[r + 1] >= u[r] + T_swap - M * (1 - next_used), name=f"slot_sequence_{r}")

# 7. 车辆对称性约束
for i in range(I - 1):
    model.addConstr(quicksum(z[i, j] for j in range(J)) >= quicksum(z[i + 1, j] for j in range(J)),
                    name=f"vehicle_symmetry_{i}")

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