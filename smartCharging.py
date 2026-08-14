import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CHARGINGTIME = 8  # 换电时间
TRIPTIME = 30  # 行程时间
NUM_VEHICLES = 7  # 车辆数量
SIMULATION_TIME = 360  # 仿真时间，单位分钟
MAX_QUEUE_LENGTH = 1  # 最大排队长度


class Battery:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.charge = capacity  # 初始满电

    def discharge(self, amount):
        self.charge -= amount

    def charge_battery(self, rate, time):
        self.charge = min(self.capacity, self.charge + rate * time)

    def __lt__(self, other):
        """定义小于运算符，使 Battery 实例可以被 heapq 排序"""
        return self.charge < other.charge


class Vehicle:
    def __init__(self, vehicle_id, battery, high_battery_threshold, low_battery_threshold, alpha, dec):
        self.id = vehicle_id
        self.battery = battery
        self.running_time = 0
        self.high_battery_threshold = high_battery_threshold
        self.low_battery_threshold = low_battery_threshold
        self.state = "running"  # 运行、换电中、排队中、前往换电站
        self.wait_time = 0  # 记录等待时间
        self.trip_end_time = 0  # 记录当前行程结束时间
        self.travel_end_time = 0  # 记录前往换电站的结束时间
        self.alpha = alpha
        self.dec = dec

    def start_trip(self, current_time):
        """开始行程，设置 30 分钟后结束"""
        if self.battery.charge >= self.low_battery_threshold:
            self.trip_end_time = current_time + TRIPTIME
            self.state = "in_trip"
        else:
            assert False

    def end_trip(self, current_time, swap_queue_length, max_length):
        """结束行程，判断是否需要去换电站"""
        self.battery.discharge(10)
        self.running_time += TRIPTIME

        if self.needs_swap(swap_queue_length, max_length):
            self.state = "traveling_to_station"
            # 如果换电站排队中有车辆，节约 10 分钟
            travel_time = 10
            self.travel_end_time = current_time + travel_time
        else:
            self.state = "running"

    def needs_swap(self, swap_queue_length, max_length):
        """判断车辆是否需要换电"""
        if self.battery.charge >= self.high_battery_threshold:
            return False
        elif self.battery.charge < self.low_battery_threshold:
            return True
        elif swap_queue_length > max_length:
            return False
        else:
            soc_factor = (self.high_battery_threshold - self.battery.charge) / (self.high_battery_threshold - self.low_battery_threshold)
            # 队列因子（范围：0~1，队列越长值越小）
            queue_factor = 1 - (swap_queue_length / max_length)
            # 加权综合得分
            score = self.alpha * soc_factor + (1 - self.alpha) * queue_factor
            return score >= self.dec


class BatterySwapStation:
    def __init__(self, num_batteries=10, charge_time=90):
        self.charge_rate = 100 / charge_time  # 充电速率
        self.available_batteries = [Battery() for _ in range(num_batteries)]
        self.swap_queue = []
        self.traveling_to_station_queue = []
        self.last_swap_end_time = 0  # 记录最近一次换电完成的时间

    # def add_to_charging(self, battery):
    #     """添加电池到充电队列"""
    #     heapq.heappush(self.charging_batteries, (battery.charge, battery))
    def add_to_charging(self, battery):
        """添加电池到充电队列"""
        self.available_batteries.append(battery)

    def charge_batteries(self, time_step=1):
        """每个时间步，所有电池都均匀充电"""
        for i in range(len(self.available_batteries)):
            battery = self.available_batteries[i]
            battery.charge_battery(self.charge_rate, time_step)

    def swap_battery(self, vehicle, swap_ready_threshold, current_time):
        """处理换电，确保两次换电至少相差 8 分钟，并随机选择一块符合条件的电池"""
        # 确保前一个换电完成至少 8 分钟后才允许换电
        if current_time < self.last_swap_end_time:
            return False  # 不能换电，等待时间未满足

        # 选择所有符合 swap_ready_threshold 的电池
        eligible_batteries = [battery for battery in self.available_batteries if
                              battery.charge >= swap_ready_threshold]

        if eligible_batteries:
            # 随机选择一块电池
            # new_battery = random.choice(eligible_batteries)
            # 选电量最多的
            tt = sorted([(b.charge, b) for b in eligible_batteries], reverse=True, key=lambda x: x[0])
            new_battery = tt[0][1]
            self.available_batteries.remove(new_battery)

            # 车辆换下的电池入充电队列
            self.add_to_charging(vehicle.battery)
            vehicle.battery = new_battery

            # 记录换电完成时间
            self.last_swap_end_time = current_time + CHARGINGTIME  # 8 分钟换电
            return True


def simulate(high_battery_threshold, low_battery_threshold, swap_ready_threshold, alpha, dec):
    vehicles = [Vehicle(i, Battery(), high_battery_threshold, low_battery_threshold, alpha, dec) for i in range(NUM_VEHICLES)]
    station = BatterySwapStation()
    total_time_record = []
    queue_len = []
    battery_charing = []

    time = 0
    max_length = MAX_QUEUE_LENGTH

    while time < SIMULATION_TIME:

        # if time < 120:
        #     for i in range(len(vehicles)):
        #         vehicles[i].low_battery_threshold = 80
        #     max_length = MAX_QUEUE_LENGTH
        # else:
        #     for i in range(len(vehicles)):
        #         vehicles[i].low_battery_threshold = low_battery_threshold
        #     max_length = 100

        for i in range(len(vehicles)):
            vehicles[i].low_battery_threshold = low_battery_threshold

        for vehicle in vehicles:
            if vehicle.state == "running":
                vehicle.start_trip(time)

            elif vehicle.state == "in_trip":
                if time >= vehicle.trip_end_time:
                    vehicle.end_trip(time, len(station.swap_queue)+len(station.traveling_to_station_queue), max_length)
                    if vehicle.state == "traveling_to_station" and vehicle not in station.traveling_to_station_queue:
                        station.traveling_to_station_queue.append(vehicle)

            elif vehicle.state == "traveling_to_station":
                if time >= vehicle.travel_end_time:
                    vehicle.state = "waiting"
                    station.traveling_to_station_queue.remove(vehicle)
                    station.swap_queue.append((time, vehicle))

            elif vehicle.state == "waiting":
                if station.swap_queue and station.swap_queue[0][1] == vehicle:
                    if station.swap_battery(vehicle, swap_ready_threshold, time):
                        station.swap_queue.pop(0)
                        vehicle.state = "swapping"
                        vehicle.wait_end_time = time + 10 + CHARGINGTIME  # 8min 换电 + 10min 前往起点

            elif vehicle.state == "swapping":
                if time >= vehicle.wait_end_time:
                    vehicle.state = "running"

        # 充电站电池充电
        station.charge_batteries()
        time += 1  # 时间推进 1 分钟
        total_time_record.append(sum([v.running_time for v in vehicles]))  # 运行时间占比
        queue_len.append(len(station.swap_queue))
        battery_charing.append(len([val for val in station.available_batteries if val.charge < 100]))

    return sum([v.running_time for v in vehicles]), total_time_record, queue_len, battery_charing


# 运行仿真
swap_ready_threshold = 100  # 充电站电池最少要充到 100 才能用

# aaa = []
# for low_threshold in range(35, swap_ready_threshold + 5, 5):
#     total_runtime, _a, __b, ___c = simulate(low_threshold, swap_ready_threshold)
#     aaa.append(total_runtime)
#     print(f"{low_threshold} - 总运行时间：", total_runtime, "分钟")
#     print(f"{low_threshold} - 占比：", total_runtime / (SIMULATION_TIME * NUM_VEHICLES) * 100)
#     print(aaa)

# total_runtime, queue, battery = simulate_24h(35, swap_ready_threshold)

high_threshold = 90
low_threshold = 35
max_res = -np.inf

# # form it as a matrix
# res_arr = [[0 for _ in range(21)] for __ in range(21)]
# for i in range(0, 21, 1):
#     for j in range(0, 21, 1):
#         alpha = i / 20
#         dec = j / 20
#         total_runtime, _a, __b, ___c = simulate(high_threshold, low_threshold, swap_ready_threshold, alpha, dec)
#         print(f"ALPHA={alpha}, DEC={dec} - 总运行时间：", total_runtime, "分钟")
#         print(f"ALPHA={alpha}, DEC={dec} - 占比：", total_runtime / (SIMULATION_TIME * NUM_VEHICLES) * 100)
#         res_arr[i][j] = total_runtime
#         if total_runtime > max_res:
#             max_res = total_runtime
#             best_alpha = [alpha]
#             best_dec = [dec]
#         elif total_runtime == max_res:
#             best_alpha.append(alpha)
#             best_dec.append(dec)
#
# # transform into dataframe
# df = pd.DataFrame(res_arr)


alpha = 0.9
dec = 0.6
total_runtime, _a, __b, ___c = simulate(high_threshold, low_threshold, swap_ready_threshold, alpha, dec)
print(f"{low_threshold} - 总运行时间：", total_runtime, "分钟")
print(f"{low_threshold} - 占比：", total_runtime / (SIMULATION_TIME * NUM_VEHICLES) * 100)
# draw pics
plt.plot(__b, label="queue length")
plt.plot(___c, label="charging battery")
plt.xlabel("Time (minutes)")
plt.legend()
plt.show()
