"""
ALNS
"""

import json
import math
import os
import pickle
import random
import time
from bisect import insort_right
import matplotlib.pyplot as plt

# ============================================================
# 1. 基本参数
# ============================================================

I = 24
H = 360
T_run = 35
T_swap = 10
T_to_station = T_from_station = 10
C_init = C_swap = 80
Delta = 10
C_min = 25

# 派生参数
J = H // T_run
K = (C_swap - C_min) // Delta
swapExtraTime = T_to_station + T_swap + T_from_station   # 换电额外时间
earliestSwap = T_run + T_to_station   # 换电站最早换电时间
latestSwap = H - T_swap - T_from_station - T_run   # 换电站最晚换电时间
stationCapacity = (latestSwap - earliestSwap) // T_swap + 1   # 换电站最大换电次数

# ============================================================
# 2. 启发式参数
# ============================================================

SEED = 42

# 每个任务数最多保留的单车模式数
MAX_PATTERNS_PER_TRIP = 250

# 贪心插入时，每个任务数最多实际评价的模式数
MAX_PATTERNS_EVALUATED = 120

# 初始随机化贪心构造次数
GREEDY_RESTARTS = 10
GREEDY_RCL_SIZE = 5

# ALNS搜索与停止条件
ALNS_ITERATIONS = 5000
ALNS_MAX_NO_IMPROVEMENT = 1000

# 每轮随机改变破坏规模，增加邻域多样性
ALNS_DESTROY_FRACTION_MIN = 0.10
ALNS_DESTROY_FRACTION_MAX = 0.30
ALNS_DESTROY_MIN = 2
ALNS_DESTROY_MAX = 12
ALNS_TIME_CLUSTER_RADIUS = 4.0   # 以T_swap为单位

# 修复参数
ALNS_REPAIR_RCL_SIZE = 3

# 1-step look-ahead：
# 当前车辆先从“最高可行任务数”层中保留若干较优pattern，
# 对每个pattern暂时插入后，再估计下一辆同质车辆能够达到的最高任务数。
ALNS_LOOKAHEAD_CURRENT_EVAL = 80       # 当前层最多实际评价的pattern数
ALNS_LOOKAHEAD_CANDIDATES = 6          # 当前最高可行trips层最多进入look-ahead的候选数
ALNS_LOOKAHEAD_NEXT_EVAL = 40          # 每个分支中，下一辆车每个trips层最多评价的pattern数

# 自适应权重
ALNS_SEGMENT_LENGTH = 100
ALNS_REACTION_FACTOR = 0.20
ALNS_SCORE_GLOBAL_BEST = 8.0
ALNS_SCORE_IMPROVE_CURRENT = 4.0
ALNS_SCORE_ACCEPTED = 1.0
ALNS_MIN_OPERATOR_WEIGHT = 0.10

# 模拟退火接受准则：标量目标的单位大致是“趟”
ALNS_INITIAL_TEMPERATURE = 0.75
ALNS_COOLING_RATE = 0.9995
ALNS_MIN_TEMPERATURE = 0.02
ALNS_LOG_EVERY = 50

OUTPUT_DIRECTORY = "."

# ============================================================
# 3. 数据格式说明
# ============================================================
#
# pattern使用普通字典：
# {
#     "trips": 总任务数,
#     "segments": (每块电池承担的任务数),
#     "positions": (每次换电前累计完成的任务数),
#     "earliest": (每次换电最早开始时刻),
#     "latest": (每次换电最晚开始时刻)
# }
#
# event使用普通字典：
# {
#     "vehicle": 车辆编号,
#     "swap_index": 第几次换电,
#     "after_trip": 第几趟后换电,
#     "start": 开始时刻,
#     "end": 结束时刻
# }
#
# solution使用普通字典：
# {
#     "assignments": 按车辆编号保存的pattern列表,
#     "schedule": 按时间排列的event列表
# }
# ============================================================

# 返回模式换电次数
def patternSwaps(pattern):
    return len(pattern["positions"])

# 返回模式的唯一键值对，用于删除重复模式（任务数+每块电池承担的任务数）
def patternKey(pattern):
    return pattern["trips"], tuple(pattern["segments"])

# 按开始时刻、结束时刻、车辆编号排序事件
def eventSortKey(event):
    return event["start"], event["end"], event["vehicle"]

# pattern和已有event在算法中只读，因此复制解时只复制外层列表。
def copySolution(solution):
    return {
        "assignments": list(solution["assignments"]),
        "schedule": list(solution["schedule"]),
    }

# 返回解中所有任务数的总和
def totalTrips(solution):
    return sum(pattern["trips"] for pattern in solution["assignments"])

# 返回解中所有换电次数的总和
def totalSwaps(solution):
    return len(solution["schedule"])

# 返回换电站最后结束时刻
def stationMakespan(solution):
    if not solution["schedule"]:
        return 0.0
    return max(event["end"] for event in solution["schedule"])


def solutionQuality(solution):
    # 解的比较顺序： 1. 总任务数越多越好； 2. 总换电次数越少越好； 3. 换电站最后结束时刻越早越好。
    return (totalTrips(solution), -totalSwaps(solution), -stationMakespan(solution))


# ============================================================
# 4. 单车上界和车队上界
# ============================================================

# 不考虑换电站排队时，单辆车最多完成多少趟
def calculateSingleVehicleMaxTrips():
    max_trips = 0
    for n in range(1, J + 1):
        min_swaps = math.ceil(n / K) - 1
        minimum_time = n * T_run + min_swaps * swapExtraTime

        if minimum_time > H:
            break
        max_trips = n

    return max_trips


def calculateUpperBound():
    """
    返回字典：
    N                    单车最大任务数
    minSwaps             完成N趟的最少换电次数
    maxSwaps             完成N趟的最多可行换电次数
    classBounds         各换电次数类别的车辆数上界
    maxVehicles          达到N趟的车辆数安全上界B
    tripUpperBound       车队任务数安全上界
    objectiveUpperBound  目标函数安全上界
    """
    N = calculateSingleVehicleMaxTrips()

    # 不需要换电即可完成N趟
    if N <= K:
        trip_upper_bound = I * N
        return {
            "N": N,
            "minSwaps": 0,
            "maxSwaps": 0,
            "classBounds": [[0, I]],
            "maxVehicles": I,
            "tripUpperBound": trip_upper_bound,
            "objectiveUpperBound": trip_upper_bound * T_run,
        }

    min_swaps = math.ceil(N / K) - 1
    max_swaps = min(N - 1, (H - N * T_run) // swapExtraTime)

    class_bounds = []

    # 对完成N趟时的每一种可行换电次数m分别计算上界
    for m in range(min_swaps, max_swaps + 1):
        minimum_time = N * T_run + m * swapExtraTime
        waiting_slack = H - minimum_time   # 剩余时间 - 用于排队等待
        window_capacity = waiting_slack // T_swap + 1
        class_bound = I

        for r in range(1, m + 1):
            lower_position = max(r, N - (m - r + 1) * K)
            upper_position = min(r * K, N - (m - r + 1))
            position_count = max(0, upper_position - lower_position + 1)

            stage_bound = position_count * window_capacity
            class_bound = min(class_bound, stage_bound)

        class_bounds.append([m, class_bound])

    # 聚合容量松弛：
    # 在忽略具体时间冲突时，优先安排换电次数少的类别。
    remaining_services = stationCapacity
    remaining_vehicles = I
    aggregate_bound = 0

    class_bounds.sort(key=lambda item: item[0])

    for item in class_bounds:
        m = item[0]
        class_bound = item[1]

        if m == 0:
            take = min(class_bound, remaining_vehicles)
        else:
            take = min(class_bound, remaining_vehicles, remaining_services // m)

        aggregate_bound += take
        remaining_vehicles -= take
        remaining_services -= take * m

        if remaining_vehicles == 0:
            break

    class_bound_sum = sum(item[1] for item in class_bounds)
    B = min(I, class_bound_sum, aggregate_bound)
    trip_upper_bound = I * (N - 1) + B

    return {
        "N": N,
        "minSwaps": min_swaps,
        "maxSwaps": max_swaps,
        "classBounds": class_bounds,
        "maxVehicles": B,
        "tripUpperBound": trip_upper_bound,
        "objectiveUpperBound": trip_upper_bound * T_run,
    }


# ============================================================
# 5. 生成单车换电模式
# ============================================================

def generateSegments(total, parts, limit):
    # 最多返回limit个结果
    results = []
    result_keys = set()

    def addResult(values):
        key = tuple(values)   # 保证可哈希
        if key not in result_keys and len(results) < limit:
            result_keys.add(key)
            results.append(list(values))

    # prefix: 已生成的分段，remaining: 剩余的总任务数，parts_left: 还需要生成的分段数    DFS搜索
    def search(prefix, remaining, parts_left):
        if len(results) >= limit:
            return

        if parts_left == 1:
            if 1 <= remaining <= K:
                values = prefix + [remaining]
                addResult(values)
                addResult(list(reversed(values)))
            return

        low = max(1, remaining - (parts_left - 1) * K)   # 当前分段的放置任务数的最小值
        high = min(K, remaining - (parts_left - 1))      # 当前分段的放置任务数的最大值

        if low > high:
            return

        # 优先生成比较均衡的分段
        average = remaining / parts_left
        candidates = list(range(low, high + 1))
        candidates.sort(key=lambda value: (abs(value - average), value))

        for value in candidates:
            search(prefix + [value], remaining - value, parts_left - 1)
            if len(results) >= limit:
                break

    search([], total, parts)
    return results


# pattern = {
#     "trips": 42,
#     "segments": [7, 7, 7, 7, 7, 7],
#     "positions": [7, 14, 21, 28, 35],
#     "earliest": [...],
#     "latest": [...]
# }
# 把一个任务分段转换成完整的换电模式

def createPattern(trips, segments):
    swaps = len(segments) - 1
    minimum_time = trips * T_run + swaps * swapExtraTime

    if minimum_time > H:
        return None

    positions = []
    cumulative = 0

    for value in segments[:-1]:
        cumulative += value
        positions.append(cumulative)

    earliest = []
    latest = []

    for r, position in enumerate(positions, start=1):

        earliest_start = position * T_run + (r - 1) * swapExtraTime + T_to_station
        latest_start = H - (trips - position) * T_run - T_swap - T_from_station - (swaps - r) * swapExtraTime

        if earliest_start > latest_start + 1e-9:
            return None

        earliest.append(float(earliest_start))
        latest.append(float(latest_start))

    return {
        "trips": trips,
        "segments": tuple(segments),
        "positions": tuple(positions),
        "earliest": tuple(earliest),
        "latest": tuple(latest),
    }


def generatePatterns(N):
    # 保存完成任务数从1到N的所有pattern
    patterns_by_trips = [[] for _ in range(N + 1)]

    for trips in range(1, N + 1):
        min_swaps = math.ceil(trips / K) - 1
        max_swaps = min(trips - 1, (H - trips * T_run) // swapExtraTime)

        if max_swaps < min_swaps:
            continue

        # 保存完成任务数为trips的所有可能换电次数
        swap_classes = list(range(min_swaps, max_swaps + 1))
        per_class_limit = max(10, math.ceil(MAX_PATTERNS_PER_TRIP / len(swap_classes)))   # 每类换电次数的模式数量上限

        patterns = []
        seen = set()

        for swaps in swap_classes:
            parts = swaps + 1
            segment_lists = generateSegments(trips, parts, per_class_limit)

            for segments in segment_lists:
                pattern = createPattern(trips, segments)
                if pattern is None:
                    continue

                key = patternKey(pattern)
                if key not in seen:
                    seen.add(key)
                    patterns.append(pattern)

        # 按换电次数、平均程度、最大任务数排序
        patterns.sort(
            key=lambda pattern: (
                patternSwaps(pattern),
                max(pattern["segments"]) - min(pattern["segments"]),
                pattern["segments"],
            )
        )

        patterns_by_trips[trips] = patterns[:MAX_PATTERNS_PER_TRIP]

    return patterns_by_trips


# ============================================================
# 6. 换电站贪心插入
# ============================================================
# schedule: 当前换电站日程
# release: 当前车的释放时间
# latest_start: 当前车的最晚开始时间
# 寻找最早可行空隙
def findEarliestGap(schedule, release, latest_start):
    start = release

    # schedule 按 start 时间排序
    for event in schedule:
        if start + T_swap <= event["start"] + 1e-9:
            if start <= latest_start + 1e-9:
                return start  # 可以放在最前面
            return None

        if start >= event["end"] - 1e-9:
            continue

        start = event["end"]

        if start > latest_start + 1e-9:
            return None

    if start <= latest_start + 1e-9:
        return start

    return None


def tryInsertPattern(schedule, vehicle, pattern):
    # 试把一辆车的整个模式插入换电站日程。成功时返回：新日程、该车总等待时间。失败时返回None
    swaps = patternSwaps(pattern)

    if swaps == 0:
        if pattern["trips"] * T_run <= H:
            return list(schedule), 0.0
        return None

    # schedule始终按eventSortKey有序；已有event只读，所以只复制列表引用。
    trial_schedule = list(schedule)

    previous_start = None
    total_wait = 0.0

    for r in range(swaps):
        if r == 0:
            release = pattern["segments"][0] * T_run + T_to_station
        else:
            release = previous_start + T_swap + T_from_station + pattern["segments"][r] * T_run + T_to_station

        latest_start = pattern["latest"][r]
        start = findEarliestGap(trial_schedule, release, latest_start)

        if start is None:
            return None

        total_wait += max(0.0, start - release)

        event = {
            "vehicle": vehicle,
            "swap_index": r + 1,
            "after_trip": pattern["positions"][r],
            "start": start,
            "end": start + T_swap,
        }

        # 二分定位后插入，避免每加入一个事件都对整个日程重新排序。
        insort_right(trial_schedule, event, key=eventSortKey)
        previous_start = start

    final_finish = previous_start + T_swap + T_from_station + pattern["segments"][-1] * T_run

    if final_finish > H + 1e-9:
        return None

    return trial_schedule, total_wait

#  计算插入评分，包括等待时间、结束时间、换电次数
def insertionScore(old_schedule, new_schedule, pattern, added_wait):
    # 日程按开始时刻排序，且所有换电服务时长相同，因此最后一个事件的end就是makespan。
    old_end = old_schedule[-1]["end"] if old_schedule else 0.0
    new_end = new_schedule[-1]["end"] if new_schedule else 0.0

    return (added_wait, max(0.0, new_end - old_end), patternSwaps(pattern))

#  移除重复模式
def removeDuplicatePatterns(patterns):
    unique = []
    seen = set()

    for pattern in patterns:
        key = patternKey(pattern)
        if key not in seen:
            seen.add(key)
            unique.append(pattern)

    return unique


def chooseAndInsertPattern(schedule, vehicle, patterns_by_trips, rng, rcl_size, extra_patterns=None):
    # 从最高任务数开始寻找可行模式。在同一任务数下，按插入评分排序，并从前rcl_size个中随机选一个。
    max_trips = len(patterns_by_trips) - 1   # 最大任务数
    candidate_limit = max(1, rcl_size)

    for trips in range(max_trips, 0, -1):
        # 全局模式池在generatePatterns中已经去重；仅在加入额外模式时再次去重。
        if extra_patterns is None:
            patterns = patterns_by_trips[trips]
        else:
            patterns = list(patterns_by_trips[trips])
            for pattern in extra_patterns:
                if pattern["trips"] == trips:
                    patterns.append(pattern)
            patterns = removeDuplicatePatterns(patterns)

        # 保留前一半 + 随机采样后一半
        if len(patterns) > MAX_PATTERNS_EVALUATED:
            head_count = MAX_PATTERNS_EVALUATED // 2
            head = patterns[:head_count]
            remaining = patterns[head_count:]
            sample_count = MAX_PATTERNS_EVALUATED - len(head)
            sample_count = min(sample_count, len(remaining))
            patterns = head + rng.sample(remaining, sample_count)

        # 只保留当前最好的RCL候选，避免同时保存大量完整日程副本。
        candidates = []
        for pattern in patterns:
            result = tryInsertPattern(schedule, vehicle, pattern)

            if result is None:
                continue

            new_schedule = result[0]   # 插入后的新调度表
            added_wait = result[1]     # 插入后的新等待时间
            score = insertionScore(schedule, new_schedule, pattern, added_wait)

            candidates.append([score, pattern, new_schedule])
            candidates.sort(key=lambda item: item[0])
            if len(candidates) > candidate_limit:
                candidates.pop()

        if candidates:
            selected = rng.choice(candidates)
            return selected[1], selected[2]  # 返回模式、新调度表

    return None

# ============================================================
# 7. 初始解和纯启发式破坏-修复
# ============================================================
# 初始解
def constructGreedySolution(patterns_by_trips, rng, randomized):
    assignments = [None] * I
    schedule = []
    vehicles = list(range(I))

    for vehicle in vehicles:
        # rcl是候选模式列表，用于选择插入模式的候选模式
        rcl_size = GREEDY_RCL_SIZE if randomized else 1

        result = chooseAndInsertPattern(schedule, vehicle, patterns_by_trips, rng, rcl_size)

        if result is None:
            raise RuntimeError(f"车辆{vehicle}无法获得可行模式，请检查参数")

        assignments[vehicle] = result[0]
        schedule = result[1]    # 新schedule替换旧schedule

    return {"assignments": assignments, "schedule": schedule}

# ============================================================
# 7.1 破坏算子
# ============================================================

def getAlnsDestroySize(rng):
    fraction = rng.uniform(ALNS_DESTROY_FRACTION_MIN, ALNS_DESTROY_FRACTION_MAX)
    size = max(ALNS_DESTROY_MIN, math.ceil(I * fraction))
    return min(size, ALNS_DESTROY_MAX, I)


def randomizedTopSelection(ranked, size, rng, pool_factor=2):
    # 从排名靠前的候选池中随机抽取
    if size >= len(ranked):
        return sorted(ranked)

    pool_size = min(len(ranked), max(size, pool_factor * size))
    return sorted(rng.sample(ranked[:pool_size], size))


def vehicleWaitingTime(solution, vehicle):
    # 计算某辆车在换电站前累计等待时间
    pattern = solution["assignments"][vehicle]
    events = [e for e in solution["schedule"] if e["vehicle"] == vehicle]
    events.sort(key=lambda e: e["swap_index"])

    previous_start = None
    total_wait = 0.0

    for r, event in enumerate(events):
        if r == 0:
            release = pattern["segments"][0] * T_run + T_to_station
        else:
            release = (
                previous_start + T_swap + T_from_station
                + pattern["segments"][r] * T_run + T_to_station
            )
        total_wait += max(0.0, event["start"] - release)
        previous_start = event["start"]

    return total_wait


def vehicleMeanSwapTime(solution, vehicle):
    starts = [e["start"] for e in solution["schedule"] if e["vehicle"] == vehicle]
    if not starts:
        return H / 2.0
    return sum(starts) / len(starts)


def destroyRandom(solution, rng, size):
    return sorted(rng.sample(list(range(I)), size))


def destroyTimeCluster(solution, rng, size):
    # 移除换电时刻聚集在同一时间区域的车辆，专门松开站点拥堵
    if not solution["schedule"]:
        return destroyRandom(solution, rng, size)

    center = rng.choice(solution["schedule"])["start"]
    radius = ALNS_TIME_CLUSTER_RADIUS * T_swap
    related = []
    seen = set()

    for event in solution["schedule"]:
        if abs(event["start"] - center) <= radius and event["vehicle"] not in seen:
            seen.add(event["vehicle"])
            related.append(event["vehicle"])

    selected = related[:]
    rng.shuffle(selected)
    selected = selected[:size]

    if len(selected) < size:
        remaining = [v for v in range(I) if v not in selected]
        selected.extend(rng.sample(remaining, size - len(selected)))

    return sorted(selected)


def destroyHighTrip(solution, rng, size):
    ranked = sorted(
        range(I),
        key=lambda v: (
            solution["assignments"][v]["trips"],
            -patternSwaps(solution["assignments"][v]),
        ),
        reverse=True,
    )
    return randomizedTopSelection(ranked, size, rng)


def destroyLowTripManySwaps(solution, rng, size):
    # 优先释放任务少但占用换电站较多的车辆
    ranked = sorted(
        range(I),
        key=lambda v: (
            solution["assignments"][v]["trips"],
            -patternSwaps(solution["assignments"][v]),
        ),
    )
    return randomizedTopSelection(ranked, size, rng)


def destroyHighWait(solution, rng, size):
    # 优先释放等待时间长的车辆，尝试重排
    ranked = sorted(
        range(I),
        key=lambda v: vehicleWaitingTime(solution, v),
        reverse=True,
    )
    return randomizedTopSelection(ranked, size, rng)


def destroyRelated(solution, rng, size):
    # 移除与随机种子车辆特征相近的一组车辆
    vehicles = list(range(I))
    seed = rng.choice(vehicles)
    seed_pattern = solution["assignments"][seed]
    seed_trips = seed_pattern["trips"]
    seed_swaps = patternSwaps(seed_pattern)
    seed_time = vehicleMeanSwapTime(solution, seed)

    others = [v for v in vehicles if v != seed]
    others.sort(
        key=lambda v: (
            6.0 * abs(solution["assignments"][v]["trips"] - seed_trips)
            + 2.0 * abs(patternSwaps(solution["assignments"][v]) - seed_swaps)
            + abs(vehicleMeanSwapTime(solution, v) - seed_time) / max(1.0, T_swap)
        )
    )

    need = size - 1
    if need <= 0:
        return [seed]

    pool_size = min(len(others), max(need, 2 * need))
    selected = [seed] + rng.sample(others[:pool_size], need)
    return sorted(selected)


ALNS_DESTROY_OPERATORS = {
    "random": destroyRandom,
    "time_cluster": destroyTimeCluster,
    "high_trip": destroyHighTrip,
    "low_trip_many_swaps": destroyLowTripManySwaps,
    "high_wait": destroyHighWait,
    "related": destroyRelated,
}


# ============================================================
# 7.2 ALNS修复算子
# ============================================================

def makePartialSolution(solution, destroyed):
    destroyed_set = set(destroyed)
    assignments = list(solution["assignments"])

    for vehicle in destroyed:
        assignments[vehicle] = None

    schedule = [
        event for event in solution["schedule"]
        if event["vehicle"] not in destroyed_set
    ]
    return assignments, schedule


def repairSequential(solution, destroyed, patterns_by_trips, rng, order, rcl_size):
    """顺序修复框架，供greedy_best与randomized_rcl复用。"""
    assignments, schedule = makePartialSolution(solution, destroyed)

    for vehicle in order:
        # 旧pattern仅作为额外候选，避免它因为全局pattern截断而丢失。
        # 对同质车辆而言，vehicle编号本身不改变可行pattern集合。
        current_pattern = solution["assignments"][vehicle]
        result = chooseAndInsertPattern(
            schedule,
            vehicle,
            patterns_by_trips,
            rng,
            rcl_size,
            extra_patterns=[current_pattern],
        )

        if result is None:
            return None

        assignments[vehicle] = result[0]
        schedule = result[1]

    return {"assignments": assignments, "schedule": schedule}


def repairGreedyBest(solution, destroyed, patterns_by_trips, rng):
    # 修复算子1：高任务旧pattern车辆优先 + 当前最优pattern插入
    order = sorted(
        destroyed,
        key=lambda v: (
            -solution["assignments"][v]["trips"],
            patternSwaps(solution["assignments"][v]),
        ),
    )
    return repairSequential(solution, destroyed, patterns_by_trips, rng, order, rcl_size=1)


def repairRandomizedRcl(solution, destroyed, patterns_by_trips, rng):
    # 修复算子2：随机车辆顺序 + 同一最高可行trips层的RCL随机插入
    order = list(destroyed)
    rng.shuffle(order)
    return repairSequential(
        solution,
        destroyed,
        patterns_by_trips,
        rng,
        order,
        rcl_size=ALNS_REPAIR_RCL_SIZE,
    )


def deterministicPatternSubset(patterns, limit):
    """
    look-ahead专用的确定性pattern子集。

    与普通RCL中的随机采样不同，look-ahead需要公平比较多个分支；
    因此每个分支在同一trips层使用相同的pattern子集：
    前一半保留排序靠前pattern，后一半从剩余pattern中均匀抽取。
    """
    if len(patterns) <= limit:
        return list(patterns)

    limit = max(1, limit)
    head_count = max(1, limit // 2)
    head = list(patterns[:head_count])
    remaining = patterns[head_count:]
    sample_count = limit - len(head)

    if sample_count <= 0 or not remaining:
        return head

    if sample_count >= len(remaining):
        return head + list(remaining)

    if sample_count == 1:
        return head + [remaining[len(remaining) // 2]]

    # 在remaining中均匀取点，避免每个look-ahead分支因随机采样产生“伪差异”。
    indices = []
    last_index = len(remaining) - 1
    for k in range(sample_count):
        index = round(k * last_index / (sample_count - 1))
        if not indices or index != indices[-1]:
            indices.append(index)

    # round极少数情况下可能产生重复索引，顺序补足。
    if len(indices) < sample_count:
        used = set(indices)
        for index in range(len(remaining)):
            if index not in used:
                indices.append(index)
                used.add(index)
                if len(indices) >= sample_count:
                    break

    return head + [remaining[index] for index in indices[:sample_count]]


def collectHighestTripCandidates(schedule, vehicle, patterns_by_trips, eval_limit, top_limit):
    """
    从最高任务数开始寻找可行pattern。
    一旦某个trips层存在可行pattern，就只在该层保留前top_limit个候选，
    因而始终保持“总任务数优先”的目标层级。

    返回元素：(insertion_score, pattern, new_schedule)
    """
    max_trips = len(patterns_by_trips) - 1

    for trips in range(max_trips, 0, -1):
        patterns = deterministicPatternSubset(patterns_by_trips[trips], eval_limit)

        candidates = []
        for pattern in patterns:
            result = tryInsertPattern(schedule, vehicle, pattern)
            if result is None:
                continue

            new_schedule, added_wait = result
            score = insertionScore(schedule, new_schedule, pattern, added_wait)
            candidates.append((score, pattern, new_schedule))

        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[:max(1, top_limit)]

    return []


def evaluateNextVehicleBest(schedule, vehicle, patterns_by_trips, eval_limit):
    """
    在给定临时schedule下，估计“下一辆同质车辆”的最佳可插入水平。

    返回：
        bestTrips：下一辆车能够达到的最高任务数；
        bestScore：在该最高任务数层中的最佳插入评分。

    这里只做一层前瞻，不真正把下一辆车提交到当前解中。
    """
    max_trips = len(patterns_by_trips) - 1

    for trips in range(max_trips, 0, -1):
        patterns = deterministicPatternSubset(patterns_by_trips[trips], eval_limit)

        best_score = None
        for pattern in patterns:
            result = tryInsertPattern(schedule, vehicle, pattern)
            if result is None:
                continue

            new_schedule, added_wait = result
            score = insertionScore(schedule, new_schedule, pattern, added_wait)

            if best_score is None or score < best_score:
                best_score = score

        if best_score is not None:
            return trips, best_score

    return 0, (float("inf"), float("inf"), float("inf"))


def repairOneStepLookAhead(solution, destroyed, patterns_by_trips, rng):
    """
    修复算子3：1-step look-ahead（pattern-oriented）。

    被破坏车辆完全同质，因此不计算vehicle-level regret；
    vehicle编号只作为event标签，修复时按编号取一个“匿名车辆槽位”。

    每一步：
    1. 找当前最高可行trips层的若干优质pattern；
    2. 对每个候选pattern暂时插入；
    3. 在该临时schedule下，看“下一辆同质车”最高还能完成多少trips；
    4. 优先选择给下一辆车保留最高trips机会的pattern；
    5. 若前瞻trips相同，再比较下一辆车的插入代价，最后比较当前pattern代价；
    6. 提交选中的pattern，再对剩余车辆重复上述过程。

    因为当前候选全部来自同一个最高可行trips层，所以比较重点是：
    “当前同样完成这么多任务时，哪个pattern对下一辆车最友好”。
    """
    assignments, schedule = makePartialSolution(solution, destroyed)
    remaining = sorted(destroyed)

    while remaining:
        # 同质车辆：这里选哪个vehicle标签并不改变pattern可行性。
        vehicle = remaining[0]

        candidates = collectHighestTripCandidates(
            schedule,
            vehicle,
            patterns_by_trips,
            eval_limit=ALNS_LOOKAHEAD_CURRENT_EVAL,
            top_limit=ALNS_LOOKAHEAD_CANDIDATES,
        )

        if not candidates:
            return None

        if len(remaining) == 1:
            # 最后一辆没有后继车辆，直接取当前最佳插入。
            selected = candidates[0]
        else:
            next_vehicle = remaining[1]
            selected = None
            best_priority = None

            for candidate in candidates:
                current_score, current_pattern, candidate_schedule = candidate

                next_trips, next_score = evaluateNextVehicleBest(
                    candidate_schedule,
                    next_vehicle,
                    patterns_by_trips,
                    eval_limit=ALNS_LOOKAHEAD_NEXT_EVAL,
                )

                # Python tuple按字典序比较；越大越优。
                # 第一优先：下一辆车还能达到的trips越多越好。
                # 后续优先：下一辆、当前车辆的等待/makespan/swaps越小越好。
                priority = (
                    next_trips,
                    -next_score[0],
                    -next_score[1],
                    -next_score[2],
                    -current_score[0],
                    -current_score[1],
                    -current_score[2],
                )

                if best_priority is None or priority > best_priority:
                    best_priority = priority
                    selected = candidate

        _, selected_pattern, selected_schedule = selected
        assignments[vehicle] = selected_pattern
        schedule = selected_schedule
        remaining.pop(0)

    return {"assignments": assignments, "schedule": schedule}


ALNS_REPAIR_OPERATORS = {
    "greedy_best": repairGreedyBest,
    "randomized_rcl": repairRandomizedRcl,
    "one_step_look_ahead": repairOneStepLookAhead,
}


# ============================================================
# 7.3 ALNS自适应选择与接受准则
# ============================================================

def rouletteSelect(weights, rng):
    total_weight = sum(max(ALNS_MIN_OPERATOR_WEIGHT, value) for value in weights.values())
    pick = rng.random() * total_weight
    cumulative = 0.0

    for name, value in weights.items():
        cumulative += max(ALNS_MIN_OPERATOR_WEIGHT, value)
        if pick <= cumulative:
            return name

    return next(reversed(weights))


def updateOperatorWeights(weights, scores, uses):
    rho = ALNS_REACTION_FACTOR

    for name in weights:
        if uses[name] > 0:
            average_reward = scores[name] / uses[name]
            weights[name] = max(
                ALNS_MIN_OPERATOR_WEIGHT,
                (1.0 - rho) * weights[name] + rho * average_reward,
            )
        scores[name] = 0.0
        uses[name] = 0


def alnsScalarScore(solution):
    """
    仅用于模拟退火的连续标量分数。
    保证“多1趟”始终比换电次数/makespan的次级改善更重要。
    """
    swap_penalty = 0.10 * totalSwaps(solution) / max(1.0, float(stationCapacity))
    makespan_penalty = 0.01 * stationMakespan(solution) / max(1.0, float(H))
    return totalTrips(solution) - swap_penalty - makespan_penalty


def acceptAlnsCandidate(current, candidate, temperature, rng):
    if solutionQuality(candidate) > solutionQuality(current):
        return True

    delta = alnsScalarScore(candidate) - alnsScalarScore(current)
    if delta >= 0.0:
        return True

    probability = math.exp(delta / max(ALNS_MIN_TEMPERATURE, temperature))
    return rng.random() < probability


def runALNS(initial_solution, patterns_by_trips, rng):
    """运行自适应大邻域搜索，返回历史最优解与逐代记录。

    连续不改进次数仅在历史最优解严格改善时清零；修复失败、
    候选被拒绝或仅改善当前解均计为不改进。None 关闭提前停止。
    """
    checkAlnsStoppingParameters()
    no_improvement = 0
    current = copySolution(initial_solution)
    best = copySolution(initial_solution)

    destroy_weights = {name: 1.0 for name in ALNS_DESTROY_OPERATORS}
    repair_weights = {name: 1.0 for name in ALNS_REPAIR_OPERATORS}
    destroy_scores = {name: 0.0 for name in ALNS_DESTROY_OPERATORS}
    repair_scores = {name: 0.0 for name in ALNS_REPAIR_OPERATORS}
    destroy_uses = {name: 0 for name in ALNS_DESTROY_OPERATORS}
    repair_uses = {name: 0 for name in ALNS_REPAIR_OPERATORS}

    temperature = ALNS_INITIAL_TEMPERATURE

    history = {
        "iterations": [0],
        "currentTrips": [totalTrips(current)],
        "bestTrips": [totalTrips(best)],
        "temperature": [temperature],
        "destroyOperator": [None],
        "repairOperator": [None],
        "accepted": [True],
        "noImprovement": [0],
        "stopReason": "max_iterations",
        "destroyWeightHistory": {name: [(0, 1.0)] for name in ALNS_DESTROY_OPERATORS},
        "repairWeightHistory": {name: [(0, 1.0)] for name in ALNS_REPAIR_OPERATORS},
    }

    for iteration in range(ALNS_ITERATIONS):
        destroy_name = rouletteSelect(destroy_weights, rng)
        repair_name = rouletteSelect(repair_weights, rng)
        destroy_uses[destroy_name] += 1
        repair_uses[repair_name] += 1

        size = getAlnsDestroySize(rng)
        destroyed = ALNS_DESTROY_OPERATORS[destroy_name](current, rng, size)
        candidate = ALNS_REPAIR_OPERATORS[repair_name](current, destroyed, patterns_by_trips, rng)

        accepted = False
        improved_best = False
        reward = 0.0
        previous_quality = solutionQuality(current)

        if candidate is not None:
            candidate_quality = solutionQuality(candidate)
            global_best = candidate_quality > solutionQuality(best)
            current_improvement = candidate_quality > previous_quality
            accepted = acceptAlnsCandidate(current, candidate, temperature, rng)

            if accepted:
                current = candidate

                if global_best:
                    best = copySolution(candidate)
                    improved_best = True
                    reward = ALNS_SCORE_GLOBAL_BEST
                elif current_improvement:
                    reward = ALNS_SCORE_IMPROVE_CURRENT
                else:
                    reward = ALNS_SCORE_ACCEPTED

        no_improvement = 0 if improved_best else no_improvement + 1
        destroy_scores[destroy_name] += reward
        repair_scores[repair_name] += reward

        temperature = max(ALNS_MIN_TEMPERATURE, temperature * ALNS_COOLING_RATE)

        step = iteration + 1
        if step % ALNS_SEGMENT_LENGTH == 0:
            updateOperatorWeights(destroy_weights, destroy_scores, destroy_uses)
            updateOperatorWeights(repair_weights, repair_scores, repair_uses)

            for name in destroy_weights:
                history["destroyWeightHistory"][name].append((step, destroy_weights[name]))
            for name in repair_weights:
                history["repairWeightHistory"][name].append((step, repair_weights[name]))

        history["iterations"].append(step)
        history["currentTrips"].append(totalTrips(current))
        history["bestTrips"].append(totalTrips(best))
        history["temperature"].append(temperature)
        history["destroyOperator"].append(destroy_name)
        history["repairOperator"].append(repair_name)
        history["accepted"].append(accepted)
        history["noImprovement"].append(no_improvement)

        if step % ALNS_LOG_EVERY == 0 or reward == ALNS_SCORE_GLOBAL_BEST:
            candidate_trips = totalTrips(candidate) if candidate is not None else "不可行"
            print(
                f"ALNS迭代{step}/{ALNS_ITERATIONS}："
                f"破坏={destroy_name}, 修复={repair_name}, 释放车辆={len(destroyed)}, "
                f"候选={candidate_trips}, 当前={totalTrips(current)}, 最好={totalTrips(best)}, "
                f"接受={accepted}, T={temperature:.4f}, 连续不改进={no_improvement}"
            )

        if (ALNS_MAX_NO_IMPROVEMENT is not None and no_improvement >= ALNS_MAX_NO_IMPROVEMENT):
            history["stopReason"] = "max_no_improvement"
            print(f"ALNS提前停止：连续{no_improvement}代未改善历史最优解，共迭代{step}代。")
            break

    print("\nALNS最终破坏算子权重：")
    for name, weight in sorted(destroy_weights.items(), key=lambda item: item[1], reverse=True):
        print(f"  {name}: {weight:.4f}")

    print("ALNS最终修复算子权重：")
    for name, weight in sorted(repair_weights.items(), key=lambda item: item[1], reverse=True):
        print(f"  {name}: {weight:.4f}")

    return best, history


def plotAlnsHistory(history):
    """绘制并保存ALNS阶段的搬运次数迭代曲线。"""
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    figure_path = os.path.join(OUTPUT_DIRECTORY, f"alns_iterations_I_{I}_H_{H}.png",)

    plt.figure(figsize=(9, 5))
    plt.plot(history["iterations"], history["currentTrips"], linewidth=1.2, label="Current solution")
    plt.plot(history["iterations"], history["bestTrips"], linewidth=2.0, label="Best solution")
    plt.xlabel("ALNS iteration")
    plt.ylabel("Total trips")
    plt.title(f"ALNS convergence (I={I}, H={H})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")

    print(f"ALNS迭代曲线：{figure_path}")
    plt.show()
    plt.close()

    return figure_path


# ============================================================
# 8. 可行性验证、结果转换和保存
# ============================================================

# 验证解是否可行
def validateSolution(solution, raise_error=True):
    def fail(message):
        if raise_error:
            raise ValueError(message)
        return False

    schedule = sorted(solution["schedule"], key=eventSortKey)

    # 1. 检查单换电站是否存在服务重叠
    for index in range(len(schedule) - 1):
        left = schedule[index]
        right = schedule[index + 1]

        if left["start"] + T_swap > right["start"] + 1e-6:
            return fail("换电站存在服务重叠")

    # 2. 按车辆整理换电事件
    events_by_vehicle = [[] for _ in range(I)]

    for event in schedule:
        events_by_vehicle[event["vehicle"]].append(event)

    # 3. 检查每辆车的运行时间逻辑
    for vehicle in range(I):
        pattern = solution["assignments"][vehicle]
        segments = pattern["segments"]

        events = sorted(events_by_vehicle[vehicle], key=lambda event: event["swap_index"])

        previous_start = None

        for r, event in enumerate(events):

            if r == 0:
                release = segments[0] * T_run + T_to_station
            else:
                release = previous_start + T_swap + T_from_station + segments[r] * T_run + T_to_station

            if event["start"] < release - 1e-6:
                return fail(f"车辆{vehicle}第{r + 1}次换电早于到站时刻")

            previous_start = event["start"]

        # 检查车辆最终完成时刻
        if len(events) == 0:
            final_finish = pattern["trips"] * T_run
        else:
            final_finish = previous_start + T_swap + T_from_station + segments[-1] * T_run

        if final_finish > H + 1e-6:
            return fail(f"车辆{vehicle}最终任务超过规划时窗")

    return True

# 同质车辆重新编号，使任务数较多的车辆编号较小。
def normalizeVehicleLabels(solution):
    old_vehicles = list(range(I))
    old_vehicles.sort(
        key=lambda vehicle: (
            -solution["assignments"][vehicle]["trips"],       # 任务数负值，使任务数较多的车辆编号较小
            patternSwaps(solution["assignments"][vehicle]),   # 换电事件数，使换电事件数较少的车辆编号较小
            vehicle,                                          # 如果任务数和换电事件数相同，按原始编号排序
        )
    )

    mapping = {}
    new_assignments = [None] * I

    for new_vehicle, old_vehicle in enumerate(old_vehicles):
        mapping[old_vehicle] = new_vehicle
        new_assignments[new_vehicle] = solution["assignments"][old_vehicle]

    new_schedule = []

    for event in solution["schedule"]:
        new_schedule.append({
            "vehicle": mapping[event["vehicle"]],
            "swap_index": event["swap_index"],
            "after_trip": event["after_trip"],
            "start": event["start"],
            "end": event["end"],
        })

    new_schedule.sort(key=eventSortKey)

    normalized = {"assignments": new_assignments, "schedule": new_schedule}
    validateSolution(normalized)

    return normalized


def buildVehicleTimeline(vehicle, pattern, schedule):
    event_by_position = {}

    for event in schedule:
        if event["vehicle"] == vehicle:
            event_by_position[event["after_trip"]] = event

    T_values = []
    z_values = []
    x_values = [0] * J
    s_values = [None] * J

    current_time = 0.0

    for task_number in range(1, J + 1):
        current_time += T_run
        T_values.append(current_time)

        z_values.append(int(task_number <= pattern["trips"]))

        if task_number in event_by_position and task_number < J:
            event = event_by_position[task_number]
            x_values[task_number - 1] = 1
            s_values[task_number - 1] = event["start"]
            current_time = event["start"] + T_swap + T_from_station

    return T_values, z_values, x_values, s_values


def solutionToCompatibleResults(solution):
    T_results = []
    z_results = []
    x_results = []
    s_results = []
    E_results = []

    for vehicle in range(I):
        pattern = solution["assignments"][vehicle]
        timeline = buildVehicleTimeline(vehicle, pattern, solution["schedule"])

        T_values, z_values, x_values, s_values = timeline

        T_results.append(T_values)
        z_results.append(z_values)
        x_results.append(x_values)
        s_results.append(s_values)

        energy = C_init
        energy_values = []

        for j in range(J):
            if z_values[j] == 0:
                energy_values.append(None)
                continue

            energy -= Delta
            energy_values.append(energy)

            if j < J - 1 and x_values[j] == 1:
                energy = C_swap

        E_results.append(energy_values)

    sorted_schedule = sorted(solution["schedule"], key=eventSortKey)
    station_schedule = []

    for index, event in enumerate(sorted_schedule):
        station_schedule.append([
            index,
            event["vehicle"],
            event["after_trip"] - 1,
            event["start"],
            event["end"],
        ])

    return {
        "T": T_results,
        "s": s_results,
        "x": x_results,
        "E": E_results,
        "z": z_results,
        "station_schedule": station_schedule,
        "objective": totalTrips(solution) * T_run,
        "total_trips": totalTrips(solution),
    }


def saveSolution(solution, upper_bound):
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    stem = f"alns_I_{I}_H_{H}"
    pickle_path = os.path.join(OUTPUT_DIRECTORY, stem + ".pkl")
    json_path = os.path.join(OUTPUT_DIRECTORY, stem + ".json")

    compatible_results = solutionToCompatibleResults(solution)

    with open(pickle_path, "wb") as file:
        pickle.dump(compatible_results, file)

    parameters = {
        "I": I,
        "H": H,
        "T_run": T_run,
        "T_swap": T_swap,
        "T_to_station": T_to_station,
        "T_from_station": T_from_station,
        "C_init": C_init,
        "C_swap": C_swap,
        "Delta": Delta,
        "C_min": C_min,
        "J": J,
        "K": K,
    }

    assignments_json = {}

    for vehicle in range(I):
        pattern = solution["assignments"][vehicle]
        assignments_json[str(vehicle)] = {
            "trips": pattern["trips"],
            "segments": pattern["segments"],
            "swap_positions": pattern["positions"],
        }

    json_data = {
        "parameters": parameters,
        "upper_bound": upper_bound,
        "solution": {
            "total_trips": totalTrips(solution),
            "objective": totalTrips(solution) * T_run,
            "total_swaps": totalSwaps(solution),
            "station_makespan": stationMakespan(solution),
            "assignments": assignments_json,
            "station_schedule": sorted(solution["schedule"], key=eventSortKey),
        },
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(json_data, file, ensure_ascii=False, indent=2)

    return pickle_path, json_path


def printSolutionSummary(solution, upper_bound):
    gap = 0.0

    if upper_bound["tripUpperBound"] > 0:
        gap = (upper_bound["tripUpperBound"] - totalTrips(solution)) / upper_bound["tripUpperBound"]

    print("\n================ 最终结果 ================")
    print(f"单车最大任务数N：{upper_bound['N']}")
    print(f"达到N趟的车辆数上界B：{upper_bound['maxVehicles']}")
    print(f"车队任务数上界：{upper_bound['tripUpperBound']}")
    print(f"启发式总任务数：{totalTrips(solution)}")
    print(f"启发式目标值：{totalTrips(solution) * T_run}")
    print(f"相对理论上界差距：{100.0 * gap:.3f}%")
    print(f"换电总次数：{totalSwaps(solution)}")
    print(f"换电站最后完成时刻：{stationMakespan(solution)}")

    distribution = {}

    for pattern in solution["assignments"]:
        trips = pattern["trips"]
        distribution[trips] = distribution.get(trips, 0) + 1

    sorted_distribution = dict(sorted(distribution.items(), key=lambda item: item[0], reverse=True))
    print(f"车辆任务数分布：{sorted_distribution}")


# ============================================================
# 9. 主程序
# ============================================================


def checkAlnsStoppingParameters():
    """校验迭代上限；排除布尔值、浮点数和不合理的代数。"""
    if type(ALNS_ITERATIONS) is not int or ALNS_ITERATIONS < 0:
        raise ValueError("ALNS_ITERATIONS必须为非负整数")
    if ALNS_MAX_NO_IMPROVEMENT is not None and (
        type(ALNS_MAX_NO_IMPROVEMENT) is not int
        or ALNS_MAX_NO_IMPROVEMENT <= 0
    ):
        raise ValueError("ALNS_MAX_NO_IMPROVEMENT必须为正整数或None")


def checkParameters():
    checkAlnsStoppingParameters()
    if I <= 0:
        raise ValueError("车辆数I必须为正数")
    if T_run <= 0 or T_swap <= 0:
        raise ValueError("T_run和T_swap必须为正数")
    if K <= 0:
        raise ValueError("当前电池参数不能支持任何任务")
    if H < T_run:
        raise ValueError("规划时窗不足以完成第一趟任务")
    # 当前假设使用初始电池和换入电池电量相同
    if C_init != C_swap:
        raise ValueError("当前假设使用初始电池和换入电池电量相同")


def main():
    # 入参校验
    checkParameters()
    rng = random.Random(SEED)
    start_time = time.time()
    # 计算上界
    upper_bound = calculateUpperBound()
    print("================ 参数和上界 ================")
    print(f"车辆数I：{I}")
    print(f"规划时窗H：{H}")
    print(f"单块电池最大连续任务数K：{K}")
    print(f"单车最大任务数N：{upper_bound['N']}")
    print(f"可行换电次数类别及其车辆数上界：{upper_bound['classBounds']}")
    print(f"达到N趟的车辆数上界B：{upper_bound['maxVehicles']}")
    print(f"车队任务数上界：{upper_bound['tripUpperBound']}")
    print(f"目标函数上界：{upper_bound['objectiveUpperBound']}")
    print(f"ALNS最大迭代代数：{ALNS_ITERATIONS}")
    print(f"最大连续不改进代数：{ALNS_MAX_NO_IMPROVEMENT}")


    # 生成单车换电模式
    print("\n================= 生成单车换电模式 ================")
    patterns_by_trips = generatePatterns(upper_bound["N"])
    total_pattern_count = sum(len(items) for items in patterns_by_trips)
    print(f"共保留{total_pattern_count}个候选模式")

    best_solution = None

    print("\n================ 构造初始解 ================")
    for restart in range(GREEDY_RESTARTS):
        # 首次使用最优插入；其余从受限候选列表中随机选择模式
        candidate = constructGreedySolution(patterns_by_trips, rng, randomized=(restart > 0))

        validateSolution(candidate)

        if best_solution is None or solutionQuality(candidate) > solutionQuality(best_solution):
            best_solution = candidate

        print(f"贪心构造{restart + 1}/{GREEDY_RESTARTS}：任务数={totalTrips(candidate)}，换电数={totalSwaps(candidate)}")

    print("\n================ 执行ALNS（纯启发式） ================")
    best_solution, alns_history = runALNS(best_solution, patterns_by_trips, rng)
    validateSolution(best_solution)

    print(f"ALNS后：任务数={totalTrips(best_solution)}，换电数={totalSwaps(best_solution)}")

    best_solution = normalizeVehicleLabels(best_solution)
    validateSolution(best_solution)
    printSolutionSummary(best_solution, upper_bound)

    # pickle_path, json_path = saveSolution(best_solution, upper_bound)
    # print("\n================ 保存结果 ================")
    # print(f"结果文件：{pickle_path}")
    # print(f"JSON文件：{json_path}")
    print(f"总运行时间：{time.time() - start_time:.2f} 秒")

    # 全部求解结束后绘制ALNS阶段的迭代曲线。
    # plotAlnsHistory(alns_history)


if __name__ == "__main__":
    main()
