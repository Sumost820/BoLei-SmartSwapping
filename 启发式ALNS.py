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

I = 20
H = 1440
T_run = 30
T_swap = 8
T_to_station = 0
T_from_station = 20
C_init = 100
C_swap = 100
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

# ALNS（不依赖Gurobi）
ALNS_ITERATIONS = 6000

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

# 保留旧名字，避免其他函数或外部脚本引用时报错
LOCAL_SEARCH_ITERATIONS = ALNS_ITERATIONS
LOCAL_DESTROY_FRACTION = 0.20
LOCAL_DESTROY_MIN = ALNS_DESTROY_MIN
LOCAL_DESTROY_MAX = ALNS_DESTROY_MAX
LOCAL_REPAIR_RCL_SIZE = ALNS_REPAIR_RCL_SIZE

# Gurobi大邻域搜索
RUN_GUROBI_LNS = False
GUROBI_LNS_ITERATIONS = 0
GUROBI_LNS_TIME_LIMIT = 20.0
GUROBI_LNS_MIP_GAP = 0.001
GUROBI_LNS_FRACTION = 0.15
GUROBI_LNS_MIN_VEHICLES = 2
GUROBI_LNS_MAX_VEHICLES = 8
GUROBI_LNS_PATTERNS_PER_VEHICLE = 50
GUROBI_OUTPUT = False

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

# 返回事件的唯一键值对，用于删除重复事件（车辆编号+换电次数+任务数）
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
    """
    解的比较顺序：
    1. 总任务数越多越好；
    2. 总换电次数越少越好；
    3. 换电站最后结束时刻越早越好。
    """
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
        tripUpperBound = I * N
        return {
            "N": N,
            "minSwaps": 0,
            "maxSwaps": 0,
            "classBounds": [[0, I]],
            "maxVehicles": I,
            "tripUpperBound": tripUpperBound,
            "objectiveUpperBound": tripUpperBound * T_run,
        }

    minSwaps = math.ceil(N / K) - 1
    maxSwaps = min(N - 1, (H - N * T_run) // swapExtraTime)

    classBounds = []

    # 对完成N趟时的每一种可行换电次数m分别计算上界
    for m in range(minSwaps, maxSwaps + 1):
        minimumTime = N * T_run + m * swapExtraTime
        waitingSlack = H - minimumTime   # 剩余时间 - 用于排队等待
        windowCapacity = waitingSlack // T_swap + 1
        classBound = I

        for r in range(1, m + 1):
            lowerPosition = max(r, N - (m - r + 1) * K)
            upperPosition = min(r * K, N - (m - r + 1))
            positionCount = max(0, upperPosition - lowerPosition + 1)

            stageBound = positionCount * windowCapacity
            classBound = min(classBound, stageBound)

        classBounds.append([m, classBound])

    # 聚合容量松弛：
    # 在忽略具体时间冲突时，优先安排换电次数少的类别。
    remainingServices = stationCapacity
    remainingVehicles = I
    aggregateBound = 0

    classBounds.sort(key=lambda item: item[0])

    for item in classBounds:
        m = item[0]
        classBound = item[1]

        if m == 0:
            take = min(classBound, remainingVehicles)
        else:
            take = min(classBound, remainingVehicles, remainingServices // m)

        aggregateBound += take
        remainingVehicles -= take
        remainingServices -= take * m

        if remainingVehicles == 0:
            break

    classBoundSum = sum(item[1] for item in classBounds)
    B = min(I, classBoundSum, aggregateBound)
    tripUpperBound = I * (N - 1) + B

    return {
        "N": N,
        "minSwaps": minSwaps,
        "maxSwaps": maxSwaps,
        "classBounds": classBounds,
        "maxVehicles": B,
        "tripUpperBound": tripUpperBound,
        "objectiveUpperBound": tripUpperBound * T_run,
    }


# ============================================================
# 5. 生成单车换电模式
# ============================================================

def generateSegments(total, parts, limit):
    """
    生成total的有序分段：
        x1 + x2 + ... + x_parts = total
        1 <= xk <= max_part

    为控制规模，最多返回limit个结果。
    """
    results = []
    resultKeys = set()

    def addResult(values):
        key = tuple(values)   # 保证可哈希
        if key not in resultKeys and len(results) < limit:
            resultKeys.add(key)
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
    minimumTime = trips * T_run + swaps * swapExtraTime

    if minimumTime > H:
        return None
    
    positions = []
    cumulative = 0

    for value in segments[:-1]:
        cumulative += value
        positions.append(cumulative)

    earliest = []
    latest = []

    for index in range(len(positions)):
        r = index + 1
        position = positions[index]

        earliestStart = position * T_run + (r - 1) * swapExtraTime + T_to_station
        latestStart = H - (trips - position) * T_run - T_swap - T_from_station - (swaps - r) * swapExtraTime

        if earliestStart > latestStart + 1e-9:
            return None

        earliest.append(float(earliestStart))
        latest.append(float(latestStart))

    return {
        "trips": trips,
        "segments": tuple(segments),
        "positions": tuple(positions),
        "earliest": tuple(earliest),
        "latest": tuple(latest),
    }


def generatePatterns(N):
    # 保存完成任务数从1到N的所有pattern
    patternsByTrips = [[] for _ in range(N + 1)]

    for trips in range(1, N + 1):
        minSwaps = math.ceil(trips / K) - 1
        max_swaps = min(trips - 1, (H - trips * T_run) // swapExtraTime)

        if max_swaps < minSwaps:
            continue

        # 保存完成任务数为trips的所有可能换电次数
        swapClasses = list(range(minSwaps, max_swaps + 1))
        perClassLimit = max(10, math.ceil(MAX_PATTERNS_PER_TRIP / len(swapClasses)))   # MAX_PATTERNS_PER_TRIPB表示每个换电次数的pattern数量

        patterns = []
        seen = set()

        for swaps in swapClasses:
            parts = swaps + 1
            segmentLists = generateSegments(trips, parts, perClassLimit)

            for segments in segmentLists:
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

        patternsByTrips[trips] = patterns[:MAX_PATTERNS_PER_TRIP]

    return patternsByTrips


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
    """
    尝试把一辆车的整个模式插入换电站日程。
    成功时返回：新日程、该车总等待时间。
    失败时返回None。
    """
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

        latestStart = pattern["latest"][r]
        start = findEarliestGap(trial_schedule, release, latestStart)

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
    """
    从最高任务数开始寻找可行模式。
    在同一任务数下，按插入评分排序，并从前rcl_size个中随机选一个。
    """
    maxTrips = len(patterns_by_trips) - 1   # 最大任务数
    candidateLimit = max(1, rcl_size)

    for trips in range(maxTrips, 0, -1):
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
            headCount = MAX_PATTERNS_EVALUATED // 2
            head = patterns[:headCount]
            remaining = patterns[headCount:]
            sampleCount = MAX_PATTERNS_EVALUATED - len(head)
            sampleCount = min(sampleCount, len(remaining))
            patterns = head + rng.sample(remaining, sampleCount)

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
            if len(candidates) > candidateLimit:
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
    assignments = [None for _ in range(I)]
    schedule = []
    vehicles = list(range(I))   # 车辆


    for vehicle in vehicles:
        # rcl是候选模式列表，用于选择插入模式的候选模式
        if randomized:
            rcl_size = GREEDY_RCL_SIZE
        else:
            rcl_size = 1

        result = chooseAndInsertPattern(schedule, vehicle, patterns_by_trips, rng, rcl_size)

        if result is None:
            raise RuntimeError(f"车辆{vehicle}无法获得可行模式，请检查参数")

        assignments[vehicle] = result[0]
        schedule = result[1]    # 新schedule替换旧schedule

    return {"assignments": assignments, "schedule": schedule}

# 根据是否为Gurobi LNS，返回破坏车辆的大小
def getDestroySize(for_gurobi):
    if for_gurobi:
        size = max(GUROBI_LNS_MIN_VEHICLES, math.ceil(I * GUROBI_LNS_FRACTION))
        return min(size, GUROBI_LNS_MAX_VEHICLES, I)

    size = max(LOCAL_DESTROY_MIN, math.ceil(I * LOCAL_DESTROY_FRACTION))
    return min(size, LOCAL_DESTROY_MAX, I)


# Gurobi-LNS仍保留原来的4种破坏逻辑，不影响下面的纯启发式ALNS。
def selectDestroyedVehicles(solution, rng, iteration, for_gurobi):
    """供原Gurobi-LNS调用：轮流使用四种简单破坏方式。"""
    size = getDestroySize(for_gurobi)
    vehicles = list(range(I))
    method = iteration % 4

    if method == 0 or not solution["schedule"]:
        return sorted(rng.sample(vehicles, size))

    if method == 1:
        centerEvent = rng.choice(solution["schedule"])
        center = centerEvent["start"]
        radius = 3 * T_swap
        related = []
        relatedSet = set()

        for event in solution["schedule"]:
            if abs(event["start"] - center) <= radius:
                vehicle = event["vehicle"]
                if vehicle not in relatedSet:
                    relatedSet.add(vehicle)
                    related.append(vehicle)

        relatedCount = min(size, len(related))
        selected = rng.sample(related, relatedCount)

        if len(selected) < size:
            remaining = [v for v in vehicles if v not in selected]
            selected.extend(rng.sample(remaining, size - len(selected)))
        return sorted(selected)

    if method == 2:
        ranked = sorted(
            vehicles,
            key=lambda vehicle: (
                solution["assignments"][vehicle]["trips"],
                -patternSwaps(solution["assignments"][vehicle]),
            ),
            reverse=True,
        )
        poolSize = max(size, min(len(ranked), 2 * size))
        return sorted(rng.sample(ranked[:poolSize], size))

    ranked = sorted(
        vehicles,
        key=lambda vehicle: (
            solution["assignments"][vehicle]["trips"],
            -patternSwaps(solution["assignments"][vehicle]),
        ),
    )
    poolSize = max(size, min(len(ranked), 2 * size))
    return sorted(rng.sample(ranked[:poolSize], size))


# ============================================================
# 7.1 ALNS破坏算子
# ============================================================

def getAlnsDestroySize(rng):
    fraction = rng.uniform(ALNS_DESTROY_FRACTION_MIN, ALNS_DESTROY_FRACTION_MAX)
    size = max(ALNS_DESTROY_MIN, math.ceil(I * fraction))
    return min(size, ALNS_DESTROY_MAX, I)


def randomizedTopSelection(ranked, size, rng, pool_factor=2):
    """从排名靠前的候选池中随机抽取，兼顾针对性和随机性。"""
    if size >= len(ranked):
        return sorted(ranked)

    poolSize = min(len(ranked), max(size, pool_factor * size))
    return sorted(rng.sample(ranked[:poolSize], size))


def vehicleWaitingTime(solution, vehicle):
    """计算某辆车在换电站前累计等待时间。"""
    pattern = solution["assignments"][vehicle]
    events = [e for e in solution["schedule"] if e["vehicle"] == vehicle]
    events.sort(key=lambda e: e["swap_index"])

    previousStart = None
    totalWait = 0.0

    for r, event in enumerate(events):
        if r == 0:
            release = pattern["segments"][0] * T_run + T_to_station
        else:
            release = (
                previousStart + T_swap + T_from_station
                + pattern["segments"][r] * T_run + T_to_station
            )
        totalWait += max(0.0, event["start"] - release)
        previousStart = event["start"]

    return totalWait


def vehicleMeanSwapTime(solution, vehicle):
    starts = [e["start"] for e in solution["schedule"] if e["vehicle"] == vehicle]
    if not starts:
        return H / 2.0
    return sum(starts) / len(starts)


def destroyRandom(solution, rng, size):
    return sorted(rng.sample(list(range(I)), size))


def destroyTimeCluster(solution, rng, size):
    """移除换电时刻聚集在同一时间区域的车辆，专门松开站点拥堵。"""
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
    """优先释放任务少但占用换电站较多的车辆。"""
    ranked = sorted(
        range(I),
        key=lambda v: (
            solution["assignments"][v]["trips"],
            -patternSwaps(solution["assignments"][v]),
        ),
    )
    return randomizedTopSelection(ranked, size, rng)


def destroyHighWait(solution, rng, size):
    """优先释放等待时间长的车辆，尝试重排拥堵链。"""
    ranked = sorted(
        range(I),
        key=lambda v: vehicleWaitingTime(solution, v),
        reverse=True,
    )
    return randomizedTopSelection(ranked, size, rng)


def destroyRelated(solution, rng, size):
    """Shaw-style related removal：移除与随机种子车辆特征相近的一组车辆。"""
    vehicles = list(range(I))
    seed = rng.choice(vehicles)
    seedPattern = solution["assignments"][seed]
    seedTrips = seedPattern["trips"]
    seedSwaps = patternSwaps(seedPattern)
    seedTime = vehicleMeanSwapTime(solution, seed)

    others = [v for v in vehicles if v != seed]
    others.sort(
        key=lambda v: (
            6.0 * abs(solution["assignments"][v]["trips"] - seedTrips)
            + 2.0 * abs(patternSwaps(solution["assignments"][v]) - seedSwaps)
            + abs(vehicleMeanSwapTime(solution, v) - seedTime) / max(1.0, T_swap)
        )
    )

    need = size - 1
    if need <= 0:
        return [seed]

    poolSize = min(len(others), max(need, 2 * need))
    selected = [seed] + rng.sample(others[:poolSize], need)
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
    destroyedSet = set(destroyed)
    assignments = list(solution["assignments"])

    for vehicle in destroyed:
        assignments[vehicle] = None

    schedule = [
        event for event in solution["schedule"]
        if event["vehicle"] not in destroyedSet
    ]
    return assignments, schedule


def repairSequential(solution, destroyed, patterns_by_trips, rng, order, rcl_size):
    """顺序修复框架，供greedy_best与randomized_rcl复用。"""
    assignments, schedule = makePartialSolution(solution, destroyed)

    for vehicle in order:
        # 旧pattern仅作为额外候选，避免它因为全局pattern截断而丢失。
        # 对同质车辆而言，vehicle编号本身不改变可行pattern集合。
        currentPattern = solution["assignments"][vehicle]
        result = chooseAndInsertPattern(
            schedule,
            vehicle,
            patterns_by_trips,
            rng,
            rcl_size,
            extra_patterns=[currentPattern],
        )

        if result is None:
            return None

        assignments[vehicle] = result[0]
        schedule = result[1]

    return {"assignments": assignments, "schedule": schedule}


def repairGreedyBest(solution, destroyed, patterns_by_trips, rng):
    """修复算子1：沿用原版，高任务旧pattern车辆优先 + 当前最优pattern插入。"""
    order = sorted(
        destroyed,
        key=lambda v: (
            -solution["assignments"][v]["trips"],
            patternSwaps(solution["assignments"][v]),
        ),
    )
    return repairSequential(solution, destroyed, patterns_by_trips, rng, order, rcl_size=1)


def repairRandomizedRcl(solution, destroyed, patterns_by_trips, rng):
    """修复算子2：沿用原版，随机车辆顺序 + 同一最高可行trips层的RCL随机插入。"""
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
    headCount = max(1, limit // 2)
    head = list(patterns[:headCount])
    remaining = patterns[headCount:]
    sampleCount = limit - len(head)

    if sampleCount <= 0 or not remaining:
        return head

    if sampleCount >= len(remaining):
        return head + list(remaining)

    if sampleCount == 1:
        return head + [remaining[len(remaining) // 2]]

    # 在remaining中均匀取点，避免每个look-ahead分支因随机采样产生“伪差异”。
    indices = []
    lastIndex = len(remaining) - 1
    for k in range(sampleCount):
        index = round(k * lastIndex / (sampleCount - 1))
        if not indices or index != indices[-1]:
            indices.append(index)

    # round极少数情况下可能产生重复索引，顺序补足。
    if len(indices) < sampleCount:
        used = set(indices)
        for index in range(len(remaining)):
            if index not in used:
                indices.append(index)
                used.add(index)
                if len(indices) >= sampleCount:
                    break

    return head + [remaining[index] for index in indices[:sampleCount]]


def collectHighestTripCandidates(
    schedule,
    vehicle,
    patterns_by_trips,
    eval_limit,
    top_limit,
):
    """
    从最高任务数开始寻找可行pattern。
    一旦某个trips层存在可行pattern，就只在该层保留前top_limit个候选，
    因而始终保持“总任务数优先”的目标层级。

    返回元素：(insertion_score, pattern, new_schedule)
    """
    maxTrips = len(patterns_by_trips) - 1

    for trips in range(maxTrips, 0, -1):
        patterns = deterministicPatternSubset(
            patterns_by_trips[trips],
            eval_limit,
        )

        candidates = []
        for pattern in patterns:
            result = tryInsertPattern(schedule, vehicle, pattern)
            if result is None:
                continue

            newSchedule, addedWait = result
            score = insertionScore(schedule, newSchedule, pattern, addedWait)
            candidates.append((score, pattern, newSchedule))

        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[:max(1, top_limit)]

    return []


def evaluateNextVehicleBest(
    schedule,
    vehicle,
    patterns_by_trips,
    eval_limit,
):
    """
    在给定临时schedule下，估计“下一辆同质车辆”的最佳可插入水平。

    返回：
        bestTrips：下一辆车能够达到的最高任务数；
        bestScore：在该最高任务数层中的最佳插入评分。

    这里只做一层前瞻，不真正把下一辆车提交到当前解中。
    """
    maxTrips = len(patterns_by_trips) - 1

    for trips in range(maxTrips, 0, -1):
        patterns = deterministicPatternSubset(
            patterns_by_trips[trips],
            eval_limit,
        )

        bestScore = None
        for pattern in patterns:
            result = tryInsertPattern(schedule, vehicle, pattern)
            if result is None:
                continue

            newSchedule, addedWait = result
            score = insertionScore(schedule, newSchedule, pattern, addedWait)

            if bestScore is None or score < bestScore:
                bestScore = score

        if bestScore is not None:
            return trips, bestScore

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
            nextVehicle = remaining[1]
            selected = None
            bestPriority = None

            for candidate in candidates:
                currentScore, currentPattern, candidateSchedule = candidate

                nextTrips, nextScore = evaluateNextVehicleBest(
                    candidateSchedule,
                    nextVehicle,
                    patterns_by_trips,
                    eval_limit=ALNS_LOOKAHEAD_NEXT_EVAL,
                )

                # Python tuple按字典序比较；越大越优。
                # 第一优先：下一辆车还能达到的trips越多越好。
                # 后续优先：下一辆、当前车辆的等待/makespan/swaps越小越好。
                priority = (
                    nextTrips,
                    -nextScore[0],
                    -nextScore[1],
                    -nextScore[2],
                    -currentScore[0],
                    -currentScore[1],
                    -currentScore[2],
                )

                if bestPriority is None or priority > bestPriority:
                    bestPriority = priority
                    selected = candidate

        _, selectedPattern, selectedSchedule = selected
        assignments[vehicle] = selectedPattern
        schedule = selectedSchedule
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
    totalWeight = sum(max(ALNS_MIN_OPERATOR_WEIGHT, value) for value in weights.values())
    pick = rng.random() * totalWeight
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
            averageReward = scores[name] / uses[name]
            weights[name] = max(
                ALNS_MIN_OPERATOR_WEIGHT,
                (1.0 - rho) * weights[name] + rho * averageReward,
            )
        scores[name] = 0.0
        uses[name] = 0


def alnsScalarScore(solution):
    """
    仅用于模拟退火的连续标量分数。
    保证“多1趟”始终比换电次数/makespan的次级改善更重要。
    """
    swapPenalty = 0.10 * totalSwaps(solution) / max(1.0, float(stationCapacity))
    makespanPenalty = 0.01 * stationMakespan(solution) / max(1.0, float(H))
    return totalTrips(solution) - swapPenalty - makespanPenalty


def acceptAlnsCandidate(current, candidate, temperature, rng):
    if solutionQuality(candidate) > solutionQuality(current):
        return True

    delta = alnsScalarScore(candidate) - alnsScalarScore(current)
    if delta >= 0.0:
        return True

    probability = math.exp(delta / max(ALNS_MIN_TEMPERATURE, temperature))
    return rng.random() < probability


def runALNS(initial_solution, patterns_by_trips, rng):
    current = copySolution(initial_solution)
    best = copySolution(initial_solution)

    destroyWeights = {name: 1.0 for name in ALNS_DESTROY_OPERATORS}
    repairWeights = {name: 1.0 for name in ALNS_REPAIR_OPERATORS}
    destroyScores = {name: 0.0 for name in ALNS_DESTROY_OPERATORS}
    repairScores = {name: 0.0 for name in ALNS_REPAIR_OPERATORS}
    destroyUses = {name: 0 for name in ALNS_DESTROY_OPERATORS}
    repairUses = {name: 0 for name in ALNS_REPAIR_OPERATORS}

    temperature = ALNS_INITIAL_TEMPERATURE

    history = {
        "iterations": [0],
        "currentTrips": [totalTrips(current)],
        "bestTrips": [totalTrips(best)],
        "temperature": [temperature],
        "destroyOperator": [None],
        "repairOperator": [None],
        "accepted": [True],
        "destroyWeightHistory": {name: [(0, 1.0)] for name in ALNS_DESTROY_OPERATORS},
        "repairWeightHistory": {name: [(0, 1.0)] for name in ALNS_REPAIR_OPERATORS},
    }

    for iteration in range(ALNS_ITERATIONS):
        destroyName = rouletteSelect(destroyWeights, rng)
        repairName = rouletteSelect(repairWeights, rng)
        destroyUses[destroyName] += 1
        repairUses[repairName] += 1

        size = getAlnsDestroySize(rng)
        destroyed = ALNS_DESTROY_OPERATORS[destroyName](current, rng, size)
        candidate = ALNS_REPAIR_OPERATORS[repairName](
            current,
            destroyed,
            patterns_by_trips,
            rng,
        )

        accepted = False
        reward = 0.0
        previousQuality = solutionQuality(current)

        if candidate is not None:
            candidateQuality = solutionQuality(candidate)
            globalBest = candidateQuality > solutionQuality(best)
            currentImprovement = candidateQuality > previousQuality
            accepted = acceptAlnsCandidate(current, candidate, temperature, rng)

            if accepted:
                current = candidate

                if globalBest:
                    best = copySolution(candidate)
                    reward = ALNS_SCORE_GLOBAL_BEST
                elif currentImprovement:
                    reward = ALNS_SCORE_IMPROVE_CURRENT
                else:
                    reward = ALNS_SCORE_ACCEPTED

        destroyScores[destroyName] += reward
        repairScores[repairName] += reward

        temperature = max(
            ALNS_MIN_TEMPERATURE,
            temperature * ALNS_COOLING_RATE,
        )

        step = iteration + 1
        if step % ALNS_SEGMENT_LENGTH == 0:
            updateOperatorWeights(destroyWeights, destroyScores, destroyUses)
            updateOperatorWeights(repairWeights, repairScores, repairUses)

            for name in destroyWeights:
                history["destroyWeightHistory"][name].append((step, destroyWeights[name]))
            for name in repairWeights:
                history["repairWeightHistory"][name].append((step, repairWeights[name]))

        history["iterations"].append(step)
        history["currentTrips"].append(totalTrips(current))
        history["bestTrips"].append(totalTrips(best))
        history["temperature"].append(temperature)
        history["destroyOperator"].append(destroyName)
        history["repairOperator"].append(repairName)
        history["accepted"].append(accepted)

        if step % ALNS_LOG_EVERY == 0 or reward == ALNS_SCORE_GLOBAL_BEST:
            candidateTrips = totalTrips(candidate) if candidate is not None else "不可行"
            print(
                f"ALNS迭代{step}/{ALNS_ITERATIONS}："
                f"破坏={destroyName}, 修复={repairName}, 释放车辆={len(destroyed)}, "
                f"候选={candidateTrips}, 当前={totalTrips(current)}, 最好={totalTrips(best)}, "
                f"接受={accepted}, T={temperature:.4f}"
            )

    print("\nALNS最终破坏算子权重：")
    for name, weight in sorted(destroyWeights.items(), key=lambda item: item[1], reverse=True):
        print(f"  {name}: {weight:.4f}")

    print("ALNS最终修复算子权重：")
    for name, weight in sorted(repairWeights.items(), key=lambda item: item[1], reverse=True):
        print(f"  {name}: {weight:.4f}")

    return best, history


# 为旧代码保留同名入口。
def runLocalSearch(initial_solution, patterns_by_trips, rng):
    return runALNS(initial_solution, patterns_by_trips, rng)


def plotAlnsHistory(history):
    """绘制并保存ALNS阶段的搬运次数迭代曲线。"""
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    figurePath = os.path.join(
        OUTPUT_DIRECTORY,
        f"alns_iterations_I_{I}_H_{H}.png",
    )

    plt.figure(figsize=(9, 5))
    plt.plot(
        history["iterations"],
        history["currentTrips"],
        linewidth=1.2,
        label="Current solution",
    )
    plt.plot(
        history["iterations"],
        history["bestTrips"],
        linewidth=2.0,
        label="Best solution",
    )
    plt.xlabel("ALNS iteration")
    plt.ylabel("Total trips")
    plt.title(f"ALNS convergence (I={I}, H={H})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figurePath, dpi=300, bbox_inches="tight")

    print(f"ALNS迭代曲线：{figurePath}")
    plt.show()
    plt.close()

    return figurePath


# 兼容旧函数名。
def plotLocalSearchHistory(history):
    return plotAlnsHistory(history)

# ============================================================
# 8. Gurobi大邻域搜索
# ============================================================


def AddPatternToList(pattern_list, seen, pattern):
    key = patternKey(pattern)
    if key not in seen:
        seen.add(key)
        pattern_list.append(pattern)


def selectLnsCandidatePatterns(current_pattern, patterns_by_trips, N, rng):
    """为一辆自由车辆选择少量候选模式。"""
    candidates = []
    seen = set()
    AddPatternToList(candidates, seen, current_pattern)

    first_level = max(1, current_pattern["trips"] - 2)
    relevant_levels = list(range(first_level, N + 1))
    per_level = max(1, GUROBI_LNS_PATTERNS_PER_VEHICLE // max(1, len(relevant_levels)))

    for trips in relevant_levels:
        level_patterns = patterns_by_trips[trips]

        for pattern in level_patterns[:per_level]:
            AddPatternToList(candidates, seen, pattern)

        remaining = level_patterns[per_level:]
        if remaining:
            sample_count = min(per_level, len(remaining))
            for pattern in rng.sample(remaining, sample_count):
                AddPatternToList(candidates, seen, pattern)

    candidates.sort(
        key=lambda pattern: (
            -pattern["trips"],
            patternSwaps(pattern),
            max(pattern["segments"]) - min(pattern["segments"]),
        )
    )

    if len(candidates) <= GUROBI_LNS_PATTERNS_PER_VEHICLE:
        return candidates

    selected = [current_pattern]

    for pattern in candidates:
        if patternKey(pattern) == patternKey(current_pattern):
            continue

        selected.append(pattern)

        if len(selected) >= GUROBI_LNS_PATTERNS_PER_VEHICLE:
            break

    return selected

# 生成当前解的事件开始时间映射: (vehicle, swap_index) -> start_time
def currentEventStartMap(solution):
    start_map = {}
    for event in solution["schedule"]:
        key = (event["vehicle"], event["swap_index"])
        start_map[key] = event["start"]

    return start_map

# ============================================================
# 9. 可行性验证、结果转换和保存
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
    eventsByVehicle = [[] for _ in range(I)]

    for event in schedule:
        eventsByVehicle[event["vehicle"]].append(event)

    # 3. 检查每辆车的运行时间逻辑
    for vehicle in range(I):
        pattern = solution["assignments"][vehicle]
        segments = pattern["segments"]

        events = sorted(eventsByVehicle[vehicle], key=lambda event: event["swap_index"])
    
        previousStart = None

        for r in range(len(events)):
            event = events[r]

            if r == 0:
                release = segments[0] * T_run + T_to_station
            else:
                release = previousStart + T_swap + T_from_station + segments[r] * T_run + T_to_station

            if event["start"] < release - 1e-6:
                return fail(f"车辆{vehicle}第{r + 1}次换电早于到站时刻")

            previousStart = event["start"]

        # 检查车辆最终完成时刻
        if len(events) == 0:
            finalFinish = pattern["trips"] * T_run
        else:
            finalFinish =  previousStart + T_swap + T_from_station + segments[-1] * T_run
            
        if finalFinish > H + 1e-6:
            return fail(f"车辆{vehicle}最终任务超过规划时窗")

    return True

# 同质车辆重新编号，使任务数较多的车辆编号较小。
def normalizeVehicleLabels(solution):
    oldVehicles = list(range(I))
    oldVehicles.sort(
        key=lambda vehicle: (
            -solution["assignments"][vehicle]["trips"],       # 任务数负值，使任务数较多的车辆编号较小
            patternSwaps(solution["assignments"][vehicle]),   # 换电事件数，使换电事件数较少的车辆编号较小
            vehicle,                                          # 如果任务数和换电事件数相同，按原始编号排序
        )
    )

    mapping = {}
    newAssignments = [None for _ in range(I)]

    for newVehicle in range(I):
        oldVehicle = oldVehicles[newVehicle]
        mapping[oldVehicle] = newVehicle
        newAssignments[newVehicle] = solution["assignments"][oldVehicle]  

    newSchedule = []

    for event in solution["schedule"]:
        newSchedule.append({
            "vehicle": mapping[event["vehicle"]],
            "swap_index": event["swap_index"],
            "after_trip": event["after_trip"],
            "start": event["start"],
            "end": event["end"],
        })

    newSchedule.sort(key=eventSortKey)

    normalized = {"assignments": newAssignments, "schedule": newSchedule}
    validateSolution(normalized)

    return normalized


def buildVehicleTimeline(vehicle, pattern, schedule):
    event_by_position = {}

    for event in schedule:
        if event["vehicle"] == vehicle:
            event_by_position[event["after_trip"]] = event

    T_values = []
    z_values = []
    x_values = [0 for _ in range(J)]
    s_values = [None for _ in range(J)]

    current_time = 0.0

    for task_number in range(1, J + 1):
        current_time += T_run
        T_values.append(current_time)

        if task_number <= pattern["trips"]:
            z_values.append(1)
        else:
            z_values.append(0)

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

        T_values = timeline[0]
        z_values = timeline[1]
        x_values = timeline[2]
        s_values = timeline[3]

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

    for index in range(len(sorted_schedule)):
        event = sorted_schedule[index]
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
# 10. 主程序
# ============================================================


def checkParameters():
    if I <= 0: raise ValueError("车辆数I必须为正数")
    if T_run <= 0 or T_swap <= 0: raise ValueError("T_run和T_swap必须为正数")
    if K <= 0: raise ValueError("当前电池参数不能支持任何任务")
    if H < T_run: raise ValueError("规划时窗不足以完成第一趟任务")
    # 当前假设使用初始电池和换入电池电量相同
    if C_init != C_swap: raise ValueError("当前假设使用初始电池和换入电池电量相同")


def main():
    # 入参校验
    checkParameters()
    rng = random.Random(SEED)
    start_time = time.time()
    # 计算上界
    upperBound = calculateUpperBound()
    print("================ 参数和上界 ================")
    print(f"车辆数I：{I}")
    print(f"规划时窗H：{H}")
    print(f"单块电池最大连续任务数K：{K}")
    print(f"单车最大任务数N：{upperBound['N']}")
    print(f"可行换电次数类别及其车辆数上界：{upperBound['classBounds']}")
    print(f"达到N趟的车辆数上界B：{upperBound['maxVehicles']}")
    print(f"车队任务数上界：{upperBound['tripUpperBound']}")
    print(f"目标函数上界：{upperBound['objectiveUpperBound']}")


    # 生成单车换电模式
    print("\n================= 生成单车换电模式 ================")
    patternsByTrips = generatePatterns(upperBound["N"])
    totalPatternCount = sum(len(items) for items in patternsByTrips)
    print(f"共保留{totalPatternCount}个候选模式")

    best_solution = None

    print("\n================ 构造初始解 ================")
    for restart in range(GREEDY_RESTARTS):
        # 构造初始解，除了第一次外，都随机打乱模式顺序
        candidate = constructGreedySolution(patternsByTrips, rng, randomized=(restart > 0))

        validateSolution(candidate)

        if best_solution is None:
            best_solution = candidate
        elif solutionQuality(candidate) > solutionQuality(best_solution):
            best_solution = candidate

        print(f"贪心构造{restart+1}/{GREEDY_RESTARTS}：任务数={totalTrips(candidate)}，换电数={totalSwaps(candidate)}")

    print("\n================ 执行ALNS（纯启发式，无需Gurobi） ================")
    best_solution, alnsHistory = runALNS(best_solution, patternsByTrips, rng)
    validateSolution(best_solution)

    print(f"ALNS后：任务数={totalTrips(best_solution)}，换电数={totalSwaps(best_solution)}")

    best_solution = normalizeVehicleLabels(best_solution)
    validateSolution(best_solution)
    printSolutionSummary(best_solution, upperBound)

    pickle_path, json_path = saveSolution(best_solution, upperBound)
    print("\n================ 保存结果 ================")
    print(f"结果文件：{pickle_path}")
    print(f"JSON文件：{json_path}")
    print(f"总运行时间：{time.time() - start_time:.2f} 秒")

    # 全部求解结束后绘制ALNS阶段的迭代曲线。
    plotAlnsHistory(alnsHistory)


if __name__ == "__main__":
    main()