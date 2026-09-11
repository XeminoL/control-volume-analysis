import numpy as np

from field import (DEGREES, EARTH_RADIUS, contains, edge_vector,
                   outward_normal, polygon_area, resample_edge)

DEFAULT_SPACING = 5000.0
REFINEMENT_SPACINGS = (40000, 20000, 10000, 5000, 2500, 1250)

DRAWS = 2000
SPEED_SIGMA = 0.05
DIRECTION_SIGMA = 5.0
CONFIDENCE = 95.0

NEWTON_STEPS = 60
STEP_LIMIT = 0.25
CONVERGENCE = 1e-4
RANK_TOLERANCE = 1e-6
BISECTION_STEPS = 60
MAX_RADIUS_DEGREES = 10.0


class Budget:
    def __init__(self, inflow, outflow, area, samples, missing, segments):
        self.inflow = inflow
        self.outflow = outflow
        self.area = area
        self.samples = samples
        self.missing = missing
        self.segments = segments

    @property
    def net(self):
        return self.outflow - self.inflow

    @property
    def coverage(self):
        total = self.samples + self.missing
        return self.samples / total if total else 0.0

    def __repr__(self):
        return (f"Budget(net={self.net:.4g}, area={self.area / 1e6:.0f} km2, "
                f"coverage={self.coverage:.0%})")


def boundary_flux(vertices, vector_field, spacing=DEFAULT_SPACING):
    if len(vertices) < 3:
        raise ValueError("a boundary needs at least three vertices")

    ordered = _counter_clockwise(vertices)
    inflow = outflow = 0.0
    samples = missing = 0
    segments = []

    for index in range(len(ordered)):
        start = ordered[index]
        end = ordered[(index + 1) % len(ordered)]
        edge_total = 0.0

        points = resample_edge(start, end, spacing)
        for a, b in zip(points[:-1], points[1:]):
            dx, dy = edge_vector(a, b)
            nx, ny, length = outward_normal(dx, dy)
            if length == 0:
                continue

            east, north = vector_field.at(0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
            if not (np.isfinite(east) and np.isfinite(north)):
                missing += 1
                continue

            samples += 1
            contribution = (east * nx + north * ny) * length
            edge_total += contribution
            if contribution >= 0:
                outflow += contribution
            else:
                inflow -= contribution

        segments.append(edge_total)

    return Budget(inflow, outflow, abs(polygon_area(ordered)),
                  samples, missing, segments)


def circulation(vertices, vector_field, spacing=DEFAULT_SPACING):
    ordered = _counter_clockwise(vertices)
    total = 0.0
    for index in range(len(ordered)):
        start = ordered[index]
        end = ordered[(index + 1) % len(ordered)]
        points = resample_edge(start, end, spacing)
        for a, b in zip(points[:-1], points[1:]):
            dx, dy = edge_vector(a, b)
            east, north = vector_field.at(0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
            if np.isfinite(east) and np.isfinite(north):
                total += east * dx + north * dy
    return total


def interior_divergence(vertices, vector_field):
    return _interior_integral(vertices, vector_field, vector_field.divergence())


def interior_curl(vertices, vector_field):
    total, counted, _ = _interior_integral(vertices, vector_field,
                                           vector_field.curl())
    return total, counted


def _counter_clockwise(vertices):
    ordered = list(vertices)
    return ordered[::-1] if polygon_area(ordered) < 0 else ordered


def _interior_integral(vertices, vector_field, values):
    latitudes = vector_field.latitudes
    longitudes = vector_field.longitudes
    d_lat = np.abs(np.gradient(latitudes)) * DEGREES
    d_lon = np.abs(np.gradient(longitudes)) * DEGREES

    total = 0.0
    counted = skipped = 0
    for i, latitude in enumerate(latitudes):
        cos_lat = np.cos(np.radians(latitude))
        for j, longitude in enumerate(longitudes):
            if not contains(vertices, latitude, longitude):
                continue
            value = values[i, j]
            if not np.isfinite(value):
                skipped += 1
                continue
            total += value * EARTH_RADIUS ** 2 * cos_lat * d_lat[i] * d_lon[j]
            counted += 1
    return total, counted, skipped


class Spread:
    def __init__(self, nominal, samples, confidence=CONFIDENCE):
        self.nominal = nominal
        self.samples = np.asarray(samples, dtype=float)
        self.confidence = confidence

    @property
    def deviation(self):
        return float(self.samples.std())

    @property
    def relative(self):
        return abs(self.deviation / self.nominal) if self.nominal else np.inf

    @property
    def interval(self):
        tail = (100 - self.confidence) / 2
        return (float(np.percentile(self.samples, tail)),
                float(np.percentile(self.samples, 100 - tail)))

    @property
    def sign_is_certain(self):
        low, high = self.interval
        return low * high > 0

    def __repr__(self):
        low, high = self.interval
        return (f"{self.nominal:.4g} +- {self.deviation:.3g} "
                f"({self.relative:.1%}), {self.confidence:.0f}% in "
                f"[{low:.4g}, {high:.4g}]")


class PerturbedField:
    def __init__(self, base, speed_sigma, direction_sigma, generator):
        self.base = base
        self.speed_sigma = speed_sigma
        self.direction_sigma = direction_sigma
        self.generator = generator
        self.latitudes = base.latitudes
        self.longitudes = base.longitudes

    def at(self, latitude, longitude):
        east, north = self.base.at(latitude, longitude)
        if not (np.isfinite(east) and np.isfinite(north)):
            return np.nan, np.nan

        speed = np.hypot(east, north)
        bearing = np.arctan2(north, east)
        speed = max(speed + self.generator.normal(0, self.speed_sigma), 0.0)
        bearing += np.radians(self.generator.normal(0, self.direction_sigma))
        return speed * np.cos(bearing), speed * np.sin(bearing)


def measurement_spread(vertices, vector_field, draws=DRAWS,
                       speed_sigma=SPEED_SIGMA,
                       direction_sigma=DIRECTION_SIGMA, seed=0):
    generator = np.random.default_rng(seed)
    nominal = boundary_flux(vertices, vector_field).net

    samples = np.empty(draws)
    for index in range(draws):
        noisy = PerturbedField(vector_field, speed_sigma,
                               direction_sigma, generator)
        samples[index] = boundary_flux(vertices, noisy).net
    return Spread(nominal, samples)


def discretisation_error(vertices, vector_field, spacings=REFINEMENT_SPACINGS):
    values = [boundary_flux(vertices, vector_field, spacing=s).net
              for s in spacings]
    changes = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
    return list(zip(spacings, values)), (changes[-1] if changes else np.nan)


def theorem_gap(vertices, vector_field, spacing=2500):
    budget = boundary_flux(vertices, vector_field, spacing=spacing)
    interior, counted, skipped = interior_divergence(vertices, vector_field)
    absolute = abs(budget.net - interior)
    gross = max(budget.inflow, budget.outflow, 1e-12)
    return {
        "boundary": budget.net,
        "interior": interior,
        "absolute": absolute,
        "against_gross": absolute / gross,
        "cells": counted,
        "cells_missing": skipped,
        "boundary_missing": budget.missing,
    }


def summarise(vertices, vector_field, draws=DRAWS, seed=0):
    spread = measurement_spread(vertices, vector_field, draws=draws, seed=seed)
    sequence, last_change = discretisation_error(vertices, vector_field)
    return {
        "net": spread.nominal,
        "spread": spread,
        "refinement": sequence,
        "refinement_change": last_change,
        "gap": theorem_gap(vertices, vector_field),
    }


def circle(centre, radius_degrees, vertices=12):
    latitude, longitude = centre
    cos_lat = np.cos(np.radians(latitude))
    angles = np.linspace(0, 2 * np.pi, vertices, endpoint=False)
    return [(latitude + radius_degrees * np.sin(angle),
             longitude + radius_degrees * np.cos(angle) / cos_lat)
            for angle in angles]


def radius_for_area(centre, area_target, vertices=12):
    low, high = 1e-4, MAX_RADIUS_DEGREES
    for _ in range(BISECTION_STEPS):
        middle = 0.5 * (low + high)
        if abs(polygon_area(circle(centre, middle, vertices))) < area_target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def flux_at(parameters, vector_field, area_target=None, spacing=5000.0):
    latitude, longitude = parameters[0], parameters[1]
    if area_target is None:
        radius = parameters[2]
    else:
        radius = radius_for_area((latitude, longitude), area_target)
    if radius <= 0:
        return np.nan
    return boundary_flux(circle((latitude, longitude), radius),
                         vector_field, spacing=spacing).net


def steps_for(vector_field):
    d_lat, d_lon = vector_field.spacing()
    cell = max(d_lat, d_lon)
    return cell, 2 * cell


def scan(vector_field, bounds, area_target, steps=9, spacing=10000.0):
    (lat_low, lat_high), (lon_low, lon_high) = bounds
    latitudes = np.linspace(lat_low, lat_high, steps)
    longitudes = np.linspace(lon_low, lon_high, steps)

    grid = []
    for latitude in latitudes:
        row = []
        for longitude in longitudes:
            value = flux_at((latitude, longitude, 0), vector_field,
                            area_target, spacing)
            row.append(value if np.isfinite(value) else np.nan)
        grid.append(row)
    return latitudes, longitudes, np.array(grid)


def best_placement(vector_field, bounds, area_target, spacing=10000.0, steps=9):
    latitudes, longitudes, grid = scan(vector_field, bounds, area_target,
                                       steps, spacing)
    if np.all(np.isnan(grid)):
        return None

    i, j = divmod(int(np.nanargmax(np.abs(grid))), grid.shape[1])
    position = np.array([latitudes[i], longitudes[j]], dtype=float)

    def objective(parameters):
        value = flux_at(parameters, vector_field, area_target, spacing)
        return -abs(value) if np.isfinite(value) else 0.0

    gradient_step, hessian_step = steps_for(vector_field)
    history = []
    for _ in range(NEWTON_STEPS):
        gradient = _gradient(objective, position, gradient_step)
        history.append((position.copy(), objective(position),
                        float(np.linalg.norm(gradient))))
        if np.linalg.norm(gradient) < CONVERGENCE:
            break

        hessian = _hessian(objective, position, hessian_step)
        curvatures = np.linalg.eigvalsh(hessian)
        if np.any(curvatures <= 0):
            step = -gradient / max(np.max(np.abs(curvatures)), 1e-12)
        else:
            step = np.linalg.solve(hessian, -gradient)
        if not np.all(np.isfinite(step)):
            break

        length = np.linalg.norm(step)
        if length > STEP_LIMIT:
            step *= STEP_LIMIT / length
        candidate = position + step
        if objective(candidate) > objective(position):
            break
        position = candidate

    radius = radius_for_area(tuple(position), area_target)
    return {
        "centre": tuple(position),
        "radius": radius,
        "flux": flux_at((*position, radius), vector_field, area_target, spacing),
        "hessian": _hessian(objective, position, hessian_step),
        "gradient": _gradient(objective, position, gradient_step),
        "history": history,
        "scan": (latitudes, longitudes, grid),
    }


def is_determined(hessian, tolerance=RANK_TOLERANCE):
    curvatures = np.linalg.eigvalsh(hessian)
    largest = np.max(np.abs(curvatures))
    if largest == 0:
        return False, curvatures, np.inf
    smallest = np.min(np.abs(curvatures))
    return smallest / largest > tolerance, curvatures, largest / max(smallest, 1e-300)


def _gradient(objective, parameters, step):
    parameters = np.asarray(parameters, dtype=float)
    out = np.empty(parameters.size)
    for index in range(parameters.size):
        forward = parameters.copy()
        backward = parameters.copy()
        forward[index] += step
        backward[index] -= step
        out[index] = (objective(forward) - objective(backward)) / (2 * step)
    return out


def _hessian(objective, parameters, step):
    parameters = np.asarray(parameters, dtype=float)
    size = parameters.size
    out = np.empty((size, size))
    centre = objective(parameters)

    for i in range(size):
        for j in range(i, size):
            if i == j:
                forward, backward = parameters.copy(), parameters.copy()
                forward[i] += step
                backward[i] -= step
                value = (objective(forward) - 2 * centre
                         + objective(backward)) / step ** 2
            else:
                corners = []
                for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    shifted = parameters.copy()
                    shifted[i] += si * step
                    shifted[j] += sj * step
                    corners.append(objective(shifted))
                value = (corners[0] - corners[1] - corners[2]
                         + corners[3]) / (4 * step ** 2)
            out[i, j] = out[j, i] = value
    return out