from gurobipy import Model, GRB, quicksum

# 参数设置
I = 7  # 车辆数
H = 600  # 规划时窗（分钟）
T_run = 30  # 单趟任务时间
T_swap = 8  # 换电操作时间
T_trip = 20  # 往返换电站时间
C_init = 100  # 初始电量
C_swap = 100  # 换电后电量
Delta = 10  # 每趟耗电
C_min = 25  # 最低电量阈值
M = 10000  # 足够大的常数
J = H // T_run  # 最大任务次数（根据实际情况调整）

# 创建模型
model = Model("Battery_Swap_Scheduling")

# 创建变量
T = model.addVars(I, J, vtype=GRB.INTEGER, name="T")  # 完成任务j的时间
s = model.addVars(I, J, vtype=GRB.INTEGER, name="s")  # 换电开始时间
x = model.addVars(I, J, vtype=GRB.BINARY, name="x")  # 换电决策
E = model.addVars(I, J, vtype=GRB.INTEGER, name="E")  # 电量
z = model.addVars(I, J, vtype=GRB.BINARY, name="z")  # 是否执行任务

# 新增换电站排序变量
y = {}
for i in range(I):
    for j in range(J):
        for k in range(I):
            for l in range(J):
                if (i, j) != (k, l):
                    var_name = f"y_{i}_{j}_{k}_{l}"
                    y[(i, j), (k, l)] = model.addVar(vtype=GRB.BINARY, name=var_name)

# 目标函数：最大化总运行时间
model.setObjective(quicksum(z[i, j] * T_run for i in range(I) for j in range(J)), GRB.MAXIMIZE)

# 约束条件
# 1. 初始任务时间约束
for i in range(I):
    model.addConstr(T[i, 0] == T_run, name=f"initial_task_time_{i}")

# 2. 初始电量约束
for i in range(I):
    model.addConstr(E[i, 0] == C_init - Delta, name=f"initial_energy_{i}")

# 3. 电量与任务执行约束
for i in range(I):
    for j in range(1, J):
        # 出发前电量 >= C_min
        model.addConstr(E[i, j - 1] >= C_min  * z[i, j - 1], name=f"energy_before_task_{i}_{j}")

        # 电量更新逻辑
        model.addConstr(E[i, j] >= E[i, j - 1] - Delta - M * x[i, j-1], name=f"energy_update_1_{i}_{j}")
        model.addConstr(E[i, j] <= E[i, j - 1] - Delta + M * x[i, j-1], name=f"energy_update_2_{i}_{j}")
        model.addConstr(E[i, j] <= C_swap - Delta + M * (1 - x[i, j-1]), name=f"energy_update_3_{i}_{j}")
        model.addConstr(E[i, j] >= C_swap - Delta - M * (1 - x[i, j-1]), name=f"energy_update_4_{i}_{j}")

# 4. 任务时间更新约束
for i in range(I):
    for j in range(1, J):
        model.addConstr(T[i, j] >= T[i, j-1] + T_run - M * x[i, j-1], name=f"task_time_update_1_{i}_{j}")
        model.addConstr(T[i, j] <= T[i, j-1] + T_run + M * x[i, j-1], name=f"task_time_update_2_{i}_{j}")
        model.addConstr(T[i, j] >= s[i, j-1] + T_swap + T_trip + T_run - M * (1 - x[i, j-1]), name=f"task_time_update_3_{i}_{j}")
        model.addConstr(T[i, j] <= s[i, j-1] + T_swap + T_trip + T_run + M * (1 - x[i, j-1]), name=f"task_time_update_4_{i}_{j}")

# 5. 时间窗约束
for i in range(I):
    for j in range(J):
        model.addConstr(T[i, j] <= H + M * (1 - z[i, j]), name=f"time_window_{i}_{j}")

# 6. 任务执行连续性约束
for i in range(I):
    model.addConstr(z[i, 0] == 1, name=f"task_start_{i}")  # 第一个任务必须执行
    for j in range(1, J):
        model.addConstr(z[i, j] <= z[i, j - 1], name=f"task_continuity_{i}_{j}")
        model.addConstr(z[i, j] >= x[i, j], name=f"task_continuity_2_{i}_{j}")

# 7. 换电开始时间约束
for i in range(I):
    for j in range(J):
        model.addConstr(s[i, j] >= T[i, j] - M * (1 - x[i, j]), name=f"swap_start_time_{i}_{j}")

# 8. 换电站排队约束（新增）

# 8.1 排队互斥约束
for i in range(I):
    for j in range(J):
        for k in range(i, I):
            for l in range(J):
                if (i, j) != (k, l):
                    if k > i or (k == i and l > j):  # 确保不重复添加约束
                        # 获取对应的y变量
                        key1 = ((i, j), (k, l))
                        key2 = ((k, l), (i, j))
                        y_ik = y.get(key1, None)
                        y_ki = y.get(key2, None)

                        # 添加互斥约束 - 1
                        model.addConstr(
                            s[k, l] >= s[i, j] + T_swap - M * (1 - y_ik),
                            name=f"queue_constraint_{i}_{j}_{k}_{l}"
                        )

                        # 添加互斥约束 - 2
                        model.addConstr(
                            s[i, j] >= s[k, l] + T_swap - M * (1 - y_ki),
                            name=f"queue_constraint_2_{i}_{j}_{k}_{l}"
                        )

                        # 添加y的显式约束（仅当x_ij=x_kl=1时生效）
                        model.addConstr(
                            y_ik + y_ki <= x[i, j],
                            name=f"y_activation_x_{i}_{j}_{k}_{l}"
                        )
                        model.addConstr(
                            y_ik + y_ki <= x[k, l],
                            name=f"y_activation_y_{i}_{j}_{k}_{l}"
                        )
                        model.addConstr(
                            y_ik + y_ki >= x[i, j] + x[k, l] - 1,
                            name=f"y_activation_z_{i}_{j}_{k}_{l}"
                        )

# 8.2 确保仅换电事件生效
for i in range(I):
    for j in range(J):
        model.addConstr(s[i, j] <= M * x[i, j], name=f"swap_event_{i}_{j}")

# 求解设置
model.setParam('TimeLimit', 600)  # 限制求解时间（秒）
model.optimize()

# store it anyway
# T results
T_results = []
for i in range(I):
    T_results.append([T[i, j].X for j in range(J)])
# s results
s_results = []
for i in range(I):
    s_results.append([s[i, j].X for j in range(J)])
# x results
x_results = []
for i in range(I):
    x_results.append([x[i, j].X for j in range(J)])
# E results
E_results = []
for i in range(I):
    E_results.append([E[i, j].X for j in range(J)])
# z results
z_results = []
for i in range(I):
    z_results.append([z[i, j].X for j in range(J)])
# store in a dict and save it as a .pkl file
results = {
    "T": T_results,
    "s": s_results,
    "x": x_results,
    "E": E_results,
    "z": z_results
}
import pickle
with open(f"results_I_{I}_J_{J}_H_{H}.pkl", "wb") as f:
    pickle.dump(results, f)




# 结果输出
if model.status == GRB.OPTIMAL:
    print(f"总有效作业时间：{model.objVal} 分钟")

    # T results
    T_results = []
    for i in range(I):
        T_results.append([T[i, j].X for j in range(J)])
    # s results
    s_results = []
    for i in range(I):
        s_results.append([s[i, j].X for j in range(J)])
    # x results
    x_results = []
    for i in range(I):
        x_results.append([x[i, j].X for j in range(J)])
    # E results
    E_results = []
    for i in range(I):
        E_results.append([E[i, j].X for j in range(J)])
    # z results
    z_results = []
    for i in range(I):
        z_results.append([z[i, j].X for j in range(J)])

    # store in a dict and save it as a .pkl file
    results = {
        "T": T_results,
        "s": s_results,
        "x": x_results,
        "E": E_results,
        "z": z_results
    }
    import pickle
    with open(f"results_I_{I}_J_{J}_H_{H}.pkl", "wb") as f:
        pickle.dump(results, f)

else:
    print("求解失败或未找到最优解")
    print(f"总有效作业时间：{model.objVal} 分钟")
    # check all the T for vehicle 0
    for j in range(J):
        print(f"车辆0任务{j}完成时间：{T[0, j].X}")
        print(f"车辆0换电开始时间：{s[0, j].X}")
        print(f"车辆0换电决策：{x[0, j].X}")
        print(f"车辆0电量：{E[0, j].X}")
        print(f"车辆0是否执行任务：{z[0, j].X}")
