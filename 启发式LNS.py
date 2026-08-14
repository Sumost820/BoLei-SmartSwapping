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

# 纯启发式破坏-修复
LOCAL_SEARCH_ITERATIONS = 5000
LOCAL_DESTROY_FRACTION = 0.20
LOCAL_DESTROY_MIN = 2
LOCAL_DESTROY_MAX = 12
LOCAL_REPAIR_RCL_SIZE = 3

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
def getDestroySize():

    size = max(LOCAL_DESTROY_MIN, math.ceil(I * LOCAL_DESTROY_FRACTION))
    return min(size, LOCAL_DESTROY_MAX, I)


def selectDestroyedVehicles(solution, rng, iteration):
    """轮流使用四种简单破坏方式。"""
    size = getDestroySize()
    vehicles = list(range(I))
    method = iteration % 4

    # 方法1：随机车辆。
    if method == 0 or not solution["schedule"]:
        selected = rng.sample(vehicles, size)
        return sorted(selected)

    # 方法2：选择某个时间附近的车辆
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

        # 从相关车辆中随机选择要破坏的车辆
        relatedCount = min(size, len(related))
        selected = rng.sample(related, relatedCount)

        # 不足则随机补充
        if len(selected) < size:
            remaining = [v for v in vehicles if v not in selected]
            selected.extend(rng.sample(remaining, size - len(selected)))

        return sorted(selected)

    # 方法3：优先破坏高任务车辆
    if method == 2:
        # 按任务数和换电次数排序
        ranked = sorted(
            vehicles,
            key=lambda vehicle: (
                solution["assignments"][vehicle]["trips"],
                -patternSwaps(solution["assignments"][vehicle]),
            ),
            reverse=True,
        )

        poolSize = max(size, min(len(ranked), 2 * size))
        selected = rng.sample(ranked[:poolSize], size)
        return sorted(selected)

    if method == 3:
        # 方法4：优先破坏低任务、换电较多的车辆
        ranked = sorted(
            vehicles,
            key=lambda vehicle: (
                solution["assignments"][vehicle]["trips"],
                -patternSwaps(solution["assignments"][vehicle]),
            ),
        )

        poolSize = max(size, min(len(ranked), 2 * size))
        selected = rng.sample(ranked[:poolSize], size)
        return sorted(selected)

# 修复破坏的车辆
def repairDestroyedVehicles(solution, destroyed, patterns_by_trips, rng):
    destroyedSet = set(destroyed)
    assignments = list(solution["assignments"])

    # 1. 清空被破坏车辆的模式
    for vehicle in destroyed:
        assignments[vehicle] = None

    # 2. 过滤有序日程会保持原顺序；已有event不修改，不需要deepcopy。
    schedule = [
        event for event in solution["schedule"]
        if event["vehicle"] not in destroyedSet
    ]

    # 3. 按车辆编号确定修复顺序
    order = sorted(destroyed)

    # 4. 逐车修复
    for vehicle in order:
        currentPattern = solution["assignments"][vehicle]

        result = chooseAndInsertPattern(schedule, vehicle, patterns_by_trips, rng,
                                           LOCAL_REPAIR_RCL_SIZE, extra_patterns=[currentPattern])

        if result is None:
            return None

        assignments[vehicle] = result[0]
        schedule = result[1]

    return {"assignments": assignments, "schedule": schedule}

def runLocalSearch(initial_solution, patterns_by_trips, rng):
    current = copySolution(initial_solution)
    best = copySolution(initial_solution)

    # 记录第0次迭代（破坏-修复开始前）以及每轮迭代后的当前解和历史最好解。
    history = {
        "iterations": [0],
        "currentTrips": [totalTrips(current)],
        "bestTrips": [totalTrips(best)],
    }

    for iteration in range(LOCAL_SEARCH_ITERATIONS):
        # 选择要破坏的车辆
        destroyed = selectDestroyedVehicles(current, rng, iteration)
        # 修复破坏的车辆
        candidate = repairDestroyedVehicles(current, destroyed, patterns_by_trips, rng)

        if candidate is not None:
            # 评估修复后的解是否更好，或者是否相同但随机接受
            if solutionQuality(candidate) > solutionQuality(current):
                current = candidate
            elif totalTrips(candidate) == totalTrips(current):
                if rng.random() < 0.05:
                    current = candidate

            if solutionQuality(current) > solutionQuality(best):
                best = copySolution(current)

            print(
                f"本地搜索迭代{iteration + 1}/{LOCAL_SEARCH_ITERATIONS}：候选任务数={totalTrips(candidate)},"
                f"当前={totalTrips(current)}，最好={totalTrips(best)}"
                    )

        history["iterations"].append(iteration + 1)
        history["currentTrips"].append(totalTrips(current))
        history["bestTrips"].append(totalTrips(best))

    return best, history


def plotLocalSearchHistory(history):
    """绘制并保存启发式破坏-修复阶段的搬运次数迭代曲线。"""

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    figurePath = os.path.join(
        OUTPUT_DIRECTORY,
        f"local_search_iterations_I_{I}_H_{H}.png",
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
    plt.xlabel("Local search iteration")
    plt.ylabel("Total trips")
    plt.title(f"Destroy-repair convergence (I={I}, H={H})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figurePath, dpi=300, bbox_inches="tight")

    print(f"迭代曲线：{figurePath}")
    plt.show()
    plt.close()

    return figurePath


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

    stem = f"LNS_I_{I}_H_{H}"
    pickle_path = os.path.join(OUTPUT_DIRECTORY, "LNSTarget", stem + ".pkl")
    json_path = os.path.join(OUTPUT_DIRECTORY, "LNSTarget", stem + ".json")

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

    print("\n================ 执行纯启发式破坏-修复 ================")
    best_solution, localSearchHistory = runLocalSearch(best_solution, patternsByTrips, rng)
    validateSolution(best_solution)

    print(f"破坏-修复后：任务数={totalTrips(best_solution)}，换电数={totalSwaps(best_solution)}")

    best_solution = normalizeVehicleLabels(best_solution)
    validateSolution(best_solution)
    printSolutionSummary(best_solution, upperBound)

    pickle_path, json_path = saveSolution(best_solution, upperBound)
    print("\n================ 保存结果 ================")
    print(f"结果文件：{pickle_path}")
    print(f"JSON文件：{json_path}")
    print(f"总运行时间：{time.time() - start_time:.2f} 秒")

    # 全部求解结束后绘制纯启发式破坏-修复阶段的迭代曲线。
    plotLocalSearchHistory(localSearchHistory)


if __name__ == "__main__":
    main()