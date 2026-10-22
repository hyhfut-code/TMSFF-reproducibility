"""Numerical kernels for the M11 common-protocol comparison.

AV-IDM follows Sharath and Velaga (2020), Eqs. (10)--(16), with the
human-like MARS model in their Table A1.  Heading-dependent velocity
components are evaluated by vector projection into the map-matched road frame;
this is invariant to CitySim's counter-clockwise heading convention.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


PI = math.pi
TWO_PI = 2.0 * math.pi
DETECTION_RANGE_M = 100.0
LONG_ACCEL_MIN = -5.0
LONG_ACCEL_MAX = 3.0
LAT_ACCEL_MIN = -3.0
LAT_ACCEL_MAX = 3.0
VEHICLE_LENGTH_M = 5.0
ROUTE_BACK_WINDOW = 14
ROUTE_FORWARD_WINDOW = 24


@njit(cache=True)
def wrap_angle(angle: float) -> float:
    while angle > PI:
        angle -= TWO_PI
    while angle <= -PI:
        angle += TWO_PI
    return angle


@njit(cache=True)
def clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


@njit(cache=True)
def project_route(
    x: float,
    y: float,
    route_x: np.ndarray,
    route_y: np.ndarray,
    route_heading: np.ndarray,
    route_left_width: np.ndarray,
    route_right_width: np.ndarray,
    route_station: np.ndarray,
    previous_index: int,
) -> tuple[int, float, float, float, float, float]:
    count = len(route_x)
    if previous_index < 0:
        start = 0
        stop = count
    else:
        start = max(0, previous_index - ROUTE_BACK_WINDOW)
        stop = min(count, previous_index + ROUTE_FORWARD_WINDOW + 1)
    best_index = start
    best_distance = 1e30
    for index in range(start, stop):
        dx = x - route_x[index]
        dy = y - route_y[index]
        distance = dx * dx + dy * dy
        if distance < best_distance:
            best_distance = distance
            best_index = index
    beta = route_heading[best_index]
    lateral = (
        -(x - route_x[best_index]) * math.sin(beta)
        + (y - route_y[best_index]) * math.cos(beta)
    )
    return (
        best_index,
        beta,
        lateral,
        route_left_width[best_index],
        route_right_width[best_index],
        route_station[best_index],
    )


@njit(cache=True)
def ray_rectangle_distance(
    origin_x: float,
    origin_y: float,
    ray_x: float,
    ray_y: float,
    center_x: float,
    center_y: float,
    heading: float,
    length: float,
    width: float,
) -> float:
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    offset_x = origin_x - center_x
    offset_y = origin_y - center_y
    local_origin_x = cos_h * offset_x + sin_h * offset_y
    local_origin_y = -sin_h * offset_x + cos_h * offset_y
    local_ray_x = cos_h * ray_x + sin_h * ray_y
    local_ray_y = -sin_h * ray_x + cos_h * ray_y
    half_length = max(0.5 * length, 0.25)
    half_width = max(0.5 * width, 0.25)
    t_min = -1e30
    t_max = 1e30

    if abs(local_ray_x) < 1e-12:
        if local_origin_x < -half_length or local_origin_x > half_length:
            return 1e30
    else:
        t1 = (-half_length - local_origin_x) / local_ray_x
        t2 = (half_length - local_origin_x) / local_ray_x
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)

    if abs(local_ray_y) < 1e-12:
        if local_origin_y < -half_width or local_origin_y > half_width:
            return 1e30
    else:
        t1 = (-half_width - local_origin_y) / local_ray_y
        t2 = (half_width - local_origin_y) / local_ray_y
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)

    if t_max < max(t_min, 0.0):
        return 1e30
    distance = t_min if t_min >= 0.0 else t_max
    return distance if distance >= 0.0 else 1e30


@njit(cache=True)
def mars_human_like(
    front_gap: float,
    rear_gap: float,
    left_gap: float,
    right_gap: float,
    speed_factor: float,
) -> float:
    bf1 = max(0.0, speed_factor - 0.9007)
    bf2 = max(0.0, 0.9007 - speed_factor)
    bf3 = max(0.0, 11.019 - right_gap)
    bf4 = (
        max(0.0, 11.019 - right_gap)
        * max(0.0, 6.3426 - left_gap)
        * max(0.0, front_gap - 7.7171)
    )
    bf5 = bf3 * max(0.0, front_gap - 23.099)
    bf6 = bf3 * max(0.0, 23.099 - front_gap)
    bf7 = bf2 * max(0.0, 21.293 - front_gap)
    bf8 = bf4 * max(0.0, rear_gap - 7.5452)
    bf9 = bf4 * max(0.0, 7.5452 - rear_gap)
    bf10 = bf3 * max(0.0, rear_gap - 6.2715)
    bf11 = bf3 * max(0.0, 6.2715 - rear_gap)
    bf12 = bf6 * max(0.0, rear_gap - 6.3901)
    bf13 = bf6 * max(0.0, 6.3901 - rear_gap)
    bf14 = (
        max(0.0, 11.019 - right_gap)
        * max(0.0, left_gap - 6.3426)
        * max(0.0, front_gap - 21.293)
    )
    return (
        -1.284e-2
        - 6.049e-1 * bf1
        - 9.088e-2 * bf2
        + 2.293e-2 * bf3
        - 2.277e-3 * bf4
        - 3.179e-2 * bf5
        - 1.248e-3 * bf6
        + 6.761e-3 * bf7
        + 5.065e-4 * bf8
        + 5.832e-4 * bf9
        - 1.826e-3 * bf10
        - 5.852e-3 * bf11
        + 1.012e-4 * bf12
        + 3.241e-4 * bf13
        + 3.122e-3 * bf14
    )


@njit(cache=True)
def avidm_perception_and_acceleration(
    parameters: np.ndarray,
    x: float,
    y: float,
    vx: float,
    vy: float,
    beta: float,
    lateral_position: float,
    left_boundary: float,
    right_boundary: float,
    environment: np.ndarray,
) -> tuple[float, float, float, float, float, float, float, float]:
    c_front, c_rear, c_left, c_right = parameters[0:4]
    a_lat, b_lat, s0_lat, time_lat = parameters[4:8]
    a_long, desired_speed, b_long, s0_long, time_long = parameters[8:13]
    ego_speed = math.hypot(vx, vy)
    ego_heading = math.atan2(vy, vx) if ego_speed > 1e-6 else beta
    ego_long = vx * math.cos(beta) + vy * math.sin(beta)
    ego_lat = -vx * math.sin(beta) + vy * math.cos(beta)

    front_gap = DETECTION_RANGE_M
    rear_gap = DETECTION_RANGE_M
    left_gap = DETECTION_RANGE_M
    right_gap = DETECTION_RANGE_M
    relative_long_sum = 0.0
    relative_lat_sum = 0.0
    hit_count = 0

    for beam_index in range(20):
        relative_beam = TWO_PI * beam_index / 20.0
        ray_angle = ego_heading + relative_beam
        ray_x = math.cos(ray_angle)
        ray_y = math.sin(ray_angle)
        minimum_distance = DETECTION_RANGE_M + 1.0
        obstacle_speed = 0.0
        obstacle_heading = beta

        lateral_ray = -ray_x * math.sin(beta) + ray_y * math.cos(beta)
        if lateral_ray > 1e-9:
            distance = (left_boundary - lateral_position) / lateral_ray
            if 0.0 < distance <= DETECTION_RANGE_M:
                minimum_distance = distance
                obstacle_speed = 0.0
                obstacle_heading = beta
        elif lateral_ray < -1e-9:
            distance = (-right_boundary - lateral_position) / lateral_ray
            if 0.0 < distance <= DETECTION_RANGE_M:
                minimum_distance = distance
                obstacle_speed = 0.0
                obstacle_heading = beta

        for obstacle_index in range(environment.shape[0]):
            if environment[obstacle_index, 6] < 0.5:
                continue
            distance = ray_rectangle_distance(
                x,
                y,
                ray_x,
                ray_y,
                environment[obstacle_index, 0],
                environment[obstacle_index, 1],
                environment[obstacle_index, 3],
                environment[obstacle_index, 4],
                environment[obstacle_index, 5],
            )
            if 0.0 < distance < minimum_distance and distance <= DETECTION_RANGE_M:
                minimum_distance = distance
                obstacle_speed = environment[obstacle_index, 2]
                obstacle_heading = environment[obstacle_index, 3]

        if minimum_distance > DETECTION_RANGE_M:
            continue
        hit_count += 1
        relative_angle = wrap_angle(ray_angle - ego_heading)
        if math.cos(relative_angle) >= 0.0:
            effective = minimum_distance * (
                1.0 + c_front * abs(math.sin(relative_angle))
            )
            front_gap = min(front_gap, effective)
        else:
            effective = minimum_distance * (
                1.0 + c_rear * abs(math.sin(relative_angle))
            )
            rear_gap = min(rear_gap, effective)
        if math.sin(relative_angle) >= 0.0:
            effective = minimum_distance * (
                1.0 + c_left * abs(math.cos(relative_angle))
            )
            left_gap = min(left_gap, effective)
        else:
            effective = minimum_distance * (
                1.0 + c_right * abs(math.cos(relative_angle))
            )
            right_gap = min(right_gap, effective)

        obstacle_long = obstacle_speed * math.cos(obstacle_heading - beta)
        obstacle_lat = obstacle_speed * math.sin(obstacle_heading - beta)
        relative_long_sum += ego_long - obstacle_long
        relative_lat_sum += ego_lat - obstacle_lat

    if hit_count:
        relative_long = relative_long_sum / hit_count
        relative_lat = relative_lat_sum / hit_count
    else:
        relative_long = 0.0
        relative_lat = 0.0

    front_gap = max(front_gap, 0.1)
    rear_gap = max(rear_gap, 0.1)
    left_gap = max(left_gap, 0.1)
    right_gap = max(right_gap, 0.1)
    dynamic_long = s0_long + max(
        0.0,
        max(ego_long, 0.0) * time_long
        + max(ego_long, 0.0) * relative_long / (2.0 * math.sqrt(a_long * b_long)),
    )
    acceleration_long = a_long * (
        1.0
        - (max(ego_long, 0.0) / max(desired_speed, 0.1)) ** 4
        - (dynamic_long / front_gap) ** 2
        + (dynamic_long / rear_gap) ** 2
    )
    lateral_speed = abs(ego_lat)
    lateral_closing = abs(relative_lat)
    dynamic_lat = s0_lat + max(
        0.0,
        lateral_speed * time_lat
        + lateral_speed * lateral_closing / (2.0 * math.sqrt(a_lat * b_lat)),
    )
    mandatory_lateral = a_lat * (
        (dynamic_lat / right_gap) ** 2 - (dynamic_lat / left_gap) ** 2
    )
    discretionary_lateral = mars_human_like(
        front_gap,
        rear_gap,
        left_gap,
        right_gap,
        max(ego_long, 0.0) / max(desired_speed, 0.1),
    )
    acceleration_lat = mandatory_lateral + discretionary_lateral
    acceleration_long = clip(acceleration_long, LONG_ACCEL_MIN, LONG_ACCEL_MAX)
    acceleration_lat = clip(acceleration_lat, LAT_ACCEL_MIN, LAT_ACCEL_MAX)
    return (
        acceleration_long,
        acceleration_lat,
        front_gap,
        rear_gap,
        left_gap,
        right_gap,
        relative_long,
        relative_lat,
    )


# This fragment is inserted into an isolated copy of the released M11 module.
# It follows the manuscript local-frame cubic, threshold rule and the main
# validation implementation's 24-node quadrature / 32-step bisection.
SHORT_RANGE_THRESHOLD_M = 15.8
ANGLE_EPS_RAD = 0.005
GEOMETRY_EPS_M = 1e-6
CANONICAL_GAUSS_NODES, CANONICAL_GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(24)


@njit(cache=True)
def slope_angle(angle):
    wrapped = (angle + 0.5 * PI) % PI - 0.5 * PI
    return clip(wrapped, -0.5 * PI + ANGLE_EPS_RAD, 0.5 * PI - ANGLE_EPS_RAD)


@njit(cache=True)
def cubic_coefficients(follower_x, follower_y, follower_heading,
                       leader_x, leader_y, leader_heading):
    follower_heading = wrap_angle(follower_heading)
    leader_heading = wrap_angle(leader_heading)
    dx = leader_x - follower_x
    dy = leader_y - follower_y
    distance = math.hypot(dx, dy)
    if distance <= GEOMETRY_EPS_M:
        return 0., 0., 0., 0., 0., False
    gap = abs(leader_heading - follower_heading)
    if abs(0.5 * gap - 0.5 * PI) <= ANGLE_EPS_RAD or gap >= PI:
        rotation = 0.5 * (leader_heading + follower_heading) - PI
    else:
        rotation = 0.5 * (leader_heading + follower_heading)
    local_x = math.cos(rotation) * dx + math.sin(rotation) * dy
    local_y = -math.sin(rotation) * dx + math.cos(rotation) * dy
    if local_x <= GEOMETRY_EPS_M:
        rotation = math.atan2(dy, dx)
        local_x = distance
        local_y = 0.
    target = math.atan2(dy, dx) if distance <= SHORT_RANGE_THRESHOLD_M else leader_heading
    physical_start = wrap_angle(follower_heading - rotation)
    physical_target = wrap_angle(target - rotation)
    # The forward graph cannot represent a backwards directed endpoint tangent.
    # Clipping is a slope regularization, not an invertible heading transform.
    # Such a state is outside the propagation domain used in this experiment.
    bound = 0.5 * PI - ANGLE_EPS_RAD
    if abs(physical_start) > bound + 1e-12 or abs(physical_target) > bound + 1e-12:
        return local_x, 0., 0., 0., rotation, False
    a1 = math.tan(slope_angle(physical_start))
    target_slope = math.tan(slope_angle(physical_target))
    a2 = (3.0 * local_y - (2.0 * a1 + target_slope) * local_x) / (local_x * local_x)
    a3 = ((a1 + target_slope) * local_x - 2.0 * local_y) / (local_x * local_x * local_x)
    return local_x, a1, a2, a3, rotation, True


@njit(cache=True)
def cubic_arc_length(a1, a2, a3, upper):
    if upper <= 0.:
        return 0.
    half = 0.5 * upper
    total = 0.
    for index in range(24):
        x = half * (CANONICAL_GAUSS_NODES[index] + 1.)
        slope = a1 + 2. * a2 * x + 3. * a3 * x * x
        total += CANONICAL_GAUSS_WEIGHTS[index] * math.sqrt(1. + slope * slope)
    return half * total


@njit(cache=True)
def advance_cubic(a1, a2, a3, path_x, distance):
    if distance <= 0.:
        return 0., 0., math.atan(a1)
    upper = min(max(path_x, 1.), 5.)
    while cubic_arc_length(a1, a2, a3, upper) < distance:
        upper *= 1.5
        if upper > max(20., 2. * path_x):
            return math.nan, math.nan, math.nan
    lower = 0.
    for _ in range(32):
        midpoint = 0.5 * (lower + upper)
        if cubic_arc_length(a1, a2, a3, midpoint) < distance:
            lower = midpoint
        else:
            upper = midpoint
    x = 0.5 * (lower + upper)
    y = a1 * x + a2 * x * x + a3 * x * x * x
    heading_local = math.atan(a1 + 2. * a2 * x + 3. * a3 * x * x)
    return x, y, heading_local





@njit(cache=True)
def tidm_acceleration(
    parameters: np.ndarray,
    speed: float,
    leader_speed: float,
    center_arc_gap: float,
) -> float:
    a_max, b_comfort, time_gap, desired_speed, minimum_gap = parameters
    net_gap = max(center_arc_gap - VEHICLE_LENGTH_M, 0.5)
    closing_speed = speed - leader_speed
    desired_gap = minimum_gap + max(
        0.0,
        speed * time_gap
        + speed * closing_speed / (2.0 * math.sqrt(a_max * b_comfort)),
    )
    acceleration = a_max * (
        1.0
        - (speed / max(desired_speed, 0.1)) ** 4
        - (desired_gap / net_gap) ** 2
    )
    return clip(acceleration, LONG_ACCEL_MIN, LONG_ACCEL_MAX)


@njit(cache=True)
def block_objective_tmsff(
    parameters: np.ndarray,
    pair_index: int,
    start: int,
    stop: int,
    observed: np.ndarray,
    leader: np.ndarray,
    route_x: np.ndarray,
    route_y: np.ndarray,
    route_heading: np.ndarray,
    route_left_width: np.ndarray,
    route_right_width: np.ndarray,
    route_station: np.ndarray,
    dt: float,
) -> float:
    x = observed[pair_index, start, 0]
    y = observed[pair_index, start, 1]
    speed = max(observed[pair_index, start, 2], 0.0)
    heading = observed[pair_index, start, 3]
    route_index = -1
    observed_station_min = observed[pair_index, start:stop, 5].min()
    observed_station_max = observed[pair_index, start:stop, 5].max()
    squared_error_sum = 0.0
    penalty_sum = 0.0

    for time_index in range(start, stop):
        dx_error = x - observed[pair_index, time_index, 0]
        dy_error = y - observed[pair_index, time_index, 1]
        squared_error_sum += dx_error * dx_error + dy_error * dy_error
        route_index, _, _, _, _, station = project_route(
            x,
            y,
            route_x[pair_index],
            route_y[pair_index],
            route_heading[pair_index],
            route_left_width[pair_index],
            route_right_width[pair_index],
            route_station[pair_index],
            route_index,
        )
        if station < observed_station_min:
            penalty_sum += (observed_station_min - station) ** 2
        elif station > observed_station_max:
            penalty_sum += (station - observed_station_max) ** 2
        if time_index >= stop - 1:
            break

        leader_x = leader[pair_index, time_index, 0]
        leader_y = leader[pair_index, time_index, 1]
        leader_speed = max(leader[pair_index, time_index, 2], 0.0)
        leader_heading = leader[pair_index, time_index, 3]
        path_x, a1, a2, a3, rotation, valid = cubic_coefficients(
            x,
            y,
            heading,
            leader_x,
            leader_y,
            leader_heading,
        )
        if not valid:
            return 1e12
        arc_gap = cubic_arc_length(a1, a2, a3, path_x)
        acceleration = tidm_acceleration(parameters, speed, leader_speed, arc_gap)
        acceleration = max(acceleration, -speed / dt)
        travel = max(0.0, speed * dt + 0.5 * acceleration * dt * dt)
        local_x, local_y, heading_local = advance_cubic(a1, a2, a3, path_x, travel)
        cos_h = math.cos(rotation)
        sin_h = math.sin(rotation)
        x += cos_h * local_x - sin_h * local_y
        y += sin_h * local_x + cos_h * local_y
        heading = wrap_angle(rotation + heading_local)
        speed = max(0.0, speed + acceleration * dt)
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(speed)):
            return 1e12

    count = stop - start
    return (squared_error_sum + penalty_sum) / max(count, 1)


@njit(cache=True)
def block_objective_avidm(
    parameters: np.ndarray,
    pair_index: int,
    start: int,
    stop: int,
    observed: np.ndarray,
    environment: np.ndarray,
    route_x: np.ndarray,
    route_y: np.ndarray,
    route_heading: np.ndarray,
    route_left_width: np.ndarray,
    route_right_width: np.ndarray,
    route_station: np.ndarray,
    dt: float,
) -> float:
    x = observed[pair_index, start, 0]
    y = observed[pair_index, start, 1]
    speed = max(observed[pair_index, start, 2], 0.0)
    heading = observed[pair_index, start, 3]
    vx = speed * math.cos(heading)
    vy = speed * math.sin(heading)
    route_index = -1
    observed_station_min = observed[pair_index, start:stop, 5].min()
    observed_station_max = observed[pair_index, start:stop, 5].max()
    squared_error_sum = 0.0
    penalty_sum = 0.0

    for time_index in range(start, stop):
        dx_error = x - observed[pair_index, time_index, 0]
        dy_error = y - observed[pair_index, time_index, 1]
        squared_error_sum += dx_error * dx_error + dy_error * dy_error
        route_index, beta, lateral, left_width, right_width, station = project_route(
            x,
            y,
            route_x[pair_index],
            route_y[pair_index],
            route_heading[pair_index],
            route_left_width[pair_index],
            route_right_width[pair_index],
            route_station[pair_index],
            route_index,
        )
        if station < observed_station_min:
            penalty_sum += (observed_station_min - station) ** 2
        elif station > observed_station_max:
            penalty_sum += (station - observed_station_max) ** 2
        if time_index >= stop - 1:
            break

        a_long, a_lat, _, _, _, _, _, _ = avidm_perception_and_acceleration(
            parameters,
            x,
            y,
            vx,
            vy,
            beta,
            lateral,
            left_width,
            right_width,
            environment[pair_index, time_index],
        )
        old_long = vx * math.cos(beta) + vy * math.sin(beta)
        old_lat = -vx * math.sin(beta) + vy * math.cos(beta)
        new_long = max(0.0, old_long + a_long * dt)
        new_lat = old_lat + a_lat * dt
        ax = a_long * math.cos(beta) - a_lat * math.sin(beta)
        ay = a_long * math.sin(beta) + a_lat * math.cos(beta)
        x += vx * dt + 0.5 * ax * dt * dt
        y += vy * dt + 0.5 * ay * dt * dt
        vx = new_long * math.cos(beta) - new_lat * math.sin(beta)
        vy = new_long * math.sin(beta) + new_lat * math.cos(beta)
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(vx) and math.isfinite(vy)):
            return 1e12

    count = stop - start
    return (squared_error_sum + penalty_sum) / max(count, 1)


@njit(cache=True)
def individual_objective_tmsff(
    parameters: np.ndarray,
    calibration: bool,
    observed: np.ndarray,
    leader: np.ndarray,
    lengths: np.ndarray,
    split_indices: np.ndarray,
    route_x: np.ndarray,
    route_y: np.ndarray,
    route_heading: np.ndarray,
    route_left_width: np.ndarray,
    route_right_width: np.ndarray,
    route_station: np.ndarray,
    dt: float,
) -> float:
    total = 0.0
    for pair_index in range(len(lengths)):
        if calibration:
            start = 0
            stop = split_indices[pair_index]
        else:
            start = split_indices[pair_index]
            stop = lengths[pair_index]
        total += block_objective_tmsff(
            parameters,
            pair_index,
            start,
            stop,
            observed,
            leader,
            route_x,
            route_y,
            route_heading,
            route_left_width,
            route_right_width,
            route_station,
            dt,
        )
    return math.sqrt(total / len(lengths))


@njit(cache=True)
def individual_objective_avidm(
    parameters: np.ndarray,
    calibration: bool,
    observed: np.ndarray,
    environment: np.ndarray,
    lengths: np.ndarray,
    split_indices: np.ndarray,
    route_x: np.ndarray,
    route_y: np.ndarray,
    route_heading: np.ndarray,
    route_left_width: np.ndarray,
    route_right_width: np.ndarray,
    route_station: np.ndarray,
    dt: float,
) -> float:
    total = 0.0
    for pair_index in range(len(lengths)):
        if calibration:
            start = 0
            stop = split_indices[pair_index]
        else:
            start = split_indices[pair_index]
            stop = lengths[pair_index]
        total += block_objective_avidm(
            parameters,
            pair_index,
            start,
            stop,
            observed,
            environment,
            route_x,
            route_y,
            route_heading,
            route_left_width,
            route_right_width,
            route_station,
            dt,
        )
    return math.sqrt(total / len(lengths))


@njit(cache=True, parallel=True)
def evaluate_population_tmsff(
    population: np.ndarray,
    observed: np.ndarray,
    leader: np.ndarray,
    lengths: np.ndarray,
    split_indices: np.ndarray,
    route_x: np.ndarray,
    route_y: np.ndarray,
    route_heading: np.ndarray,
    route_left_width: np.ndarray,
    route_right_width: np.ndarray,
    route_station: np.ndarray,
    dt: float,
) -> np.ndarray:
    output = np.empty(len(population), dtype=np.float64)
    for index in prange(len(population)):
        output[index] = individual_objective_tmsff(
            population[index],
            True,
            observed,
            leader,
            lengths,
            split_indices,
            route_x,
            route_y,
            route_heading,
            route_left_width,
            route_right_width,
            route_station,
            dt,
        )
    return output


@njit(cache=True, parallel=True)
def evaluate_population_avidm(
    population: np.ndarray,
    observed: np.ndarray,
    environment: np.ndarray,
    lengths: np.ndarray,
    split_indices: np.ndarray,
    route_x: np.ndarray,
    route_y: np.ndarray,
    route_heading: np.ndarray,
    route_left_width: np.ndarray,
    route_right_width: np.ndarray,
    route_station: np.ndarray,
    dt: float,
) -> np.ndarray:
    output = np.empty(len(population), dtype=np.float64)
    for index in prange(len(population)):
        output[index] = individual_objective_avidm(
            population[index],
            True,
            observed,
            environment,
            lengths,
            split_indices,
            route_x,
            route_y,
            route_heading,
            route_left_width,
            route_right_width,
            route_station,
            dt,
        )
    return output


@njit(cache=True)
def trace_tmsff(
    parameters: np.ndarray,
    pair_index: int,
    start: int,
    stop: int,
    observed: np.ndarray,
    leader: np.ndarray,
    dt: float,
) -> np.ndarray:
    count = stop - start
    trace = np.zeros((count, 7), dtype=np.float64)
    x = observed[pair_index, start, 0]
    y = observed[pair_index, start, 1]
    speed = max(observed[pair_index, start, 2], 0.0)
    heading = observed[pair_index, start, 3]
    acceleration = 0.0
    for local_index in range(count):
        time_index = start + local_index
        trace[local_index, 0] = x
        trace[local_index, 1] = y
        trace[local_index, 2] = speed
        trace[local_index, 3] = heading
        trace[local_index, 4] = acceleration
        if time_index >= stop - 1:
            break
        path_x, a1, a2, a3, rotation, valid = cubic_coefficients(
            x,
            y,
            heading,
            leader[pair_index, time_index, 0],
            leader[pair_index, time_index, 1],
            leader[pair_index, time_index, 3],
        )
        if not valid:
            trace[:, :] = np.nan
            return trace
        arc_gap = cubic_arc_length(a1, a2, a3, path_x)
        acceleration = tidm_acceleration(
            parameters,
            speed,
            max(leader[pair_index, time_index, 2], 0.0),
            arc_gap,
        )
        acceleration = max(acceleration, -speed / dt)
        travel = max(0.0, speed * dt + 0.5 * acceleration * dt * dt)
        local_x, local_y, heading_local = advance_cubic(a1, a2, a3, path_x, travel)
        cos_h = math.cos(rotation)
        sin_h = math.sin(rotation)
        x += cos_h * local_x - sin_h * local_y
        y += sin_h * local_x + cos_h * local_y
        heading = wrap_angle(rotation + heading_local)
        speed = max(0.0, speed + acceleration * dt)
        trace[local_index, 5] = arc_gap
        trace[local_index, 6] = arc_gap - VEHICLE_LENGTH_M
    return trace


@njit(cache=True)
def trace_avidm(
    parameters: np.ndarray,
    pair_index: int,
    start: int,
    stop: int,
    observed: np.ndarray,
    environment: np.ndarray,
    route_x: np.ndarray,
    route_y: np.ndarray,
    route_heading: np.ndarray,
    route_left_width: np.ndarray,
    route_right_width: np.ndarray,
    route_station: np.ndarray,
    dt: float,
) -> np.ndarray:
    count = stop - start
    trace = np.zeros((count, 12), dtype=np.float64)
    x = observed[pair_index, start, 0]
    y = observed[pair_index, start, 1]
    speed = max(observed[pair_index, start, 2], 0.0)
    heading = observed[pair_index, start, 3]
    vx = speed * math.cos(heading)
    vy = speed * math.sin(heading)
    route_index = -1
    a_long = 0.0
    a_lat = 0.0
    gaps = (DETECTION_RANGE_M, DETECTION_RANGE_M, DETECTION_RANGE_M, DETECTION_RANGE_M)
    for local_index in range(count):
        time_index = start + local_index
        speed = math.hypot(vx, vy)
        heading = math.atan2(vy, vx) if speed > 1e-6 else heading
        route_index, beta, lateral, left_width, right_width, station = project_route(
            x,
            y,
            route_x[pair_index],
            route_y[pair_index],
            route_heading[pair_index],
            route_left_width[pair_index],
            route_right_width[pair_index],
            route_station[pair_index],
            route_index,
        )
        trace[local_index, 0] = x
        trace[local_index, 1] = y
        trace[local_index, 2] = speed
        trace[local_index, 3] = heading
        trace[local_index, 4] = a_long
        trace[local_index, 5] = a_lat
        trace[local_index, 6] = gaps[0]
        trace[local_index, 7] = gaps[1]
        trace[local_index, 8] = gaps[2]
        trace[local_index, 9] = gaps[3]
        trace[local_index, 10] = lateral
        trace[local_index, 11] = station
        if time_index >= stop - 1:
            break
        result = avidm_perception_and_acceleration(
            parameters,
            x,
            y,
            vx,
            vy,
            beta,
            lateral,
            left_width,
            right_width,
            environment[pair_index, time_index],
        )
        a_long, a_lat = result[0], result[1]
        gaps = (result[2], result[3], result[4], result[5])
        old_long = vx * math.cos(beta) + vy * math.sin(beta)
        old_lat = -vx * math.sin(beta) + vy * math.cos(beta)
        new_long = max(0.0, old_long + a_long * dt)
        new_lat = old_lat + a_lat * dt
        ax = a_long * math.cos(beta) - a_lat * math.sin(beta)
        ay = a_long * math.sin(beta) + a_lat * math.cos(beta)
        x += vx * dt + 0.5 * ax * dt * dt
        y += vy * dt + 0.5 * ay * dt * dt
        vx = new_long * math.cos(beta) - new_lat * math.sin(beta)
        vy = new_long * math.sin(beta) + new_lat * math.cos(beta)
    return trace
