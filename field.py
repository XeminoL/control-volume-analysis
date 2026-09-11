import numpy as np

EARTH_RADIUS = 6371008.8
DEGREES = np.pi / 180.0
FILL_MARGIN = 1e-6
FOURTH_ORDER_STENCIL = ([-2, -1, 0, 1, 2], [1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12])
MEASURED_BEST_ORDER = 4
SECOND_ORDER = 2


def metres_per_degree_lat():
    return EARTH_RADIUS * DEGREES


def edge_vector(start, end):
    lat1, lon1 = start
    lat2, lon2 = end
    if abs(lat2 - lat1) < 1e-12:
        scale = np.cos(np.radians(lat1))
    else:
        a, b = np.radians(lat1), np.radians(lat2)
        scale = (np.sin(b) - np.sin(a)) / (b - a)
    return ((lon2 - lon1) * EARTH_RADIUS * DEGREES * scale,
            (lat2 - lat1) * metres_per_degree_lat())


def outward_normal(dx, dy):
    length = np.hypot(dx, dy)
    if length == 0:
        return 0.0, 0.0, 0.0
    return dy / length, -dx / length, length


def polygon_area(vertices):
    total = 0.0
    count = len(vertices)
    for index in range(count):
        lat1, lon1 = vertices[index]
        lat2, lon2 = vertices[(index + 1) % count]
        mean_sine = 0.5 * (np.sin(np.radians(lat1)) + np.sin(np.radians(lat2)))
        total += mean_sine * (lon2 - lon1) * DEGREES
    return -EARTH_RADIUS ** 2 * total


def contains(vertices, latitude, longitude):
    inside = False
    count = len(vertices)
    for index in range(count):
        lat1, lon1 = vertices[index]
        lat2, lon2 = vertices[(index + 1) % count]
        if (lat1 > latitude) != (lat2 > latitude):
            crossing = lon1 + (latitude - lat1) * (lon2 - lon1) / (lat2 - lat1)
            if longitude < crossing:
                inside = not inside
    return inside


def resample_edge(start, end, spacing_metres):
    dx, dy = edge_vector(start, end)
    pieces = max(1, int(np.ceil(np.hypot(dx, dy) / spacing_metres)))
    return [(start[0] + k / pieces * (end[0] - start[0]),
             start[1] + k / pieces * (end[1] - start[1]))
            for k in range(pieces + 1)]


def catmull_rom_weights(offset):
    t = offset
    return 0.5 * np.array([
        t * ((2 - t) * t - 1),
        t * t * (3 * t - 5) + 2,
        t * ((4 - 3 * t) * t + 1),
        t * t * (t - 1),
    ])


def _spacing_along(spacing, values, axis):
    shape = [1] * values.ndim
    shape[axis] = spacing.size
    return spacing.reshape(shape)


def _derivative(values, axis, spacing, order=MEASURED_BEST_ORDER):
    if order == SECOND_ORDER:
        return np.gradient(values, axis=axis) / _spacing_along(spacing, values, axis)

    offsets, weights = FOURTH_ORDER_STENCIL
    reach = max(offsets)
    length = values.shape[axis]
    out = np.full_like(values, np.nan, dtype=float)

    interior = [slice(None)] * values.ndim
    interior[axis] = slice(reach, length - reach)
    interior = tuple(interior)

    out[interior] = 0.0
    for offset, weight in zip(offsets, weights):
        if weight == 0:
            continue
        source = [slice(None)] * values.ndim
        source[axis] = slice(reach + offset, length - reach + offset)
        out[interior] += weight * values[tuple(source)]
    out[interior] /= _spacing_along(spacing, values, axis)[interior]
    return out


class VectorField:
    def __init__(self, latitudes, longitudes, east, north, units="m/s"):
        self.latitudes = np.asarray(latitudes, dtype=float)
        self.longitudes = np.asarray(longitudes, dtype=float)
        self.east = np.asarray(east, dtype=float)
        self.north = np.asarray(north, dtype=float)
        self.units = units

        expected = (self.latitudes.size, self.longitudes.size)
        if self.east.shape != expected:
            raise ValueError(f"east has shape {self.east.shape}, expected {expected}")
        if self.east.shape != self.north.shape:
            raise ValueError("the two components must have the same shape")

    def __repr__(self):
        return (f"VectorField({self.latitudes.size}x{self.longitudes.size}, "
                f"{self.latitudes.min():.2f}..{self.latitudes.max():.2f}N, "
                f"{self.longitudes.min():.2f}..{self.longitudes.max():.2f}E, "
                f"{self.coverage():.0%} valid)")

    def coverage(self):
        return float(np.isfinite(self.east).mean())

    def spacing(self):
        return (float(np.abs(np.diff(self.latitudes)).mean()),
                float(np.abs(np.diff(self.longitudes)).mean()))

    def at(self, latitude, longitude):
        row = np.interp(latitude, self.latitudes,
                        np.arange(self.latitudes.size), left=np.nan, right=np.nan)
        column = np.interp(longitude, self.longitudes,
                           np.arange(self.longitudes.size), left=np.nan, right=np.nan)
        if not (np.isfinite(row) and np.isfinite(column)):
            return np.nan, np.nan

        i = min(int(np.floor(row)), self.latitudes.size - 2)
        j = min(int(np.floor(column)), self.longitudes.size - 2)
        down, across = row - i, column - j

        has_room = (1 <= i <= self.latitudes.size - 3
                    and 1 <= j <= self.longitudes.size - 3)
        blend = self._bicubic if has_room else self._bilinear
        return (float(blend(self.east, i, j, down, across)),
                float(blend(self.north, i, j, down, across)))

    @staticmethod
    def _bilinear(grid, i, j, down, across):
        return ((1 - down) * (1 - across) * grid[i, j]
                + (1 - down) * across * grid[i, j + 1]
                + down * (1 - across) * grid[i + 1, j]
                + down * across * grid[i + 1, j + 1])

    @staticmethod
    def _bicubic(grid, i, j, down, across):
        block = grid[i - 1:i + 3, j - 1:j + 3]
        if not np.all(np.isfinite(block)):
            return np.nan
        return float(catmull_rom_weights(down) @ block @ catmull_rom_weights(across))

    def divergence(self, order=MEASURED_BEST_ORDER):
        cos_lat = np.cos(np.radians(self.latitudes))[:, None]
        east = _derivative(self.east, 1,
                           np.gradient(self.longitudes) * DEGREES, order)
        north = _derivative(self.north * cos_lat, 0,
                            np.gradient(self.latitudes) * DEGREES, order)
        return (east + north) / (EARTH_RADIUS * cos_lat)

    def curl(self, order=MEASURED_BEST_ORDER):
        cos_lat = np.cos(np.radians(self.latitudes))[:, None]
        north = _derivative(self.north, 1,
                            np.gradient(self.longitudes) * DEGREES, order)
        east = _derivative(self.east * cos_lat, 0,
                           np.gradient(self.latitudes) * DEGREES, order)
        return (north - east) / (EARTH_RADIUS * cos_lat)


def from_hycom(path):
    import h5py

    with h5py.File(path, "r") as handle:
        latitudes = handle["lat"][:]
        longitudes = handle["lon"][:]
        components = []
        for name in ("water_u", "water_v"):
            variable = handle[name]
            raw = np.squeeze(variable[:]).astype(float)
            fill = variable.attrs.get("_FillValue")
            if fill is not None:
                raw[np.isclose(raw, float(fill[0]), rtol=FILL_MARGIN)] = np.nan
            raw *= float(variable.attrs.get("scale_factor", [1.0])[0])
            raw += float(variable.attrs.get("add_offset", [0.0])[0])
            components.append(raw)

    return VectorField(latitudes, longitudes, *components, units="m/s")