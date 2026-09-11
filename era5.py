import urllib.request

import numpy as np

from field import VectorField

BUCKET = "https://nsf-ncar-era5.s3.amazonaws.com"
MONTHLY = ("e5.oper.an.vinteg/{month}/e5.oper.an.vinteg.162_{code}_{name}"
           ".ll025sc.{month}0100_{month}{last}23.nc")
LAST_DAY = {"01": "31", "02": "28", "03": "31", "04": "30", "05": "31",
            "06": "30", "07": "31", "08": "31", "09": "30", "10": "31",
            "11": "30", "12": "31"}
VARIABLES = {"viwve": "071", "viwvn": "072", "viwvd": "084"}

BLOCK = 4 * 1024 * 1024
TIMEOUT = 120
MAX_CACHE = 64


class RangeReader:
    def __init__(self, url, block=BLOCK, timeout=TIMEOUT):
        self.url = url
        self.block = block
        self.timeout = timeout
        self.position = 0
        self.blocks = {}
        self.requests = 0
        self.bytes_fetched = 0
        self.size = self._head()

    def _head(self):
        request = urllib.request.Request(self.url, method="HEAD")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            if response.headers.get("Accept-Ranges") != "bytes":
                raise OSError(f"{self.url} does not serve byte ranges")
            return int(response.headers["Content-Length"])

    def _block(self, index):
        if index in self.blocks:
            return self.blocks[index]

        start = index * self.block
        stop = min(start + self.block, self.size) - 1
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{stop}"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = response.read()

        self.requests += 1
        self.bytes_fetched += len(data)
        if len(self.blocks) >= MAX_CACHE:
            self.blocks.pop(next(iter(self.blocks)))
        self.blocks[index] = data
        return data

    def read(self, amount=-1):
        if amount is None or amount < 0:
            amount = self.size - self.position
        amount = min(amount, self.size - self.position)
        if amount <= 0:
            return b""

        chunks = []
        remaining = amount
        while remaining > 0:
            index = self.position // self.block
            offset = self.position - index * self.block
            piece = self._block(index)[offset:offset + remaining]
            if not piece:
                break
            chunks.append(piece)
            self.position += len(piece)
            remaining -= len(piece)
        return b"".join(chunks)

    def seek(self, offset, whence=0):
        if whence == 0:
            self.position = offset
        elif whence == 1:
            self.position += offset
        else:
            self.position = self.size + offset
        return self.position

    def tell(self):
        return self.position

    def close(self):
        self.blocks.clear()

    def __repr__(self):
        return (f"RangeReader({self.size / 1e6:.0f} MB, {self.requests} requests, "
                f"{self.bytes_fetched / 1e6:.1f} MB fetched)")


def open_remote_hdf5(url, block=BLOCK):
    import h5py

    reader = RangeReader(url, block=block)
    return h5py.File(reader, "r"), reader


def url_for(name, month):
    return (f"{BUCKET}/{MONTHLY.format(month=month, code=VARIABLES[name], name=name, last=LAST_DAY[month[4:6]])}")


def load(name, month, bounds, hour=0):
    handle, reader = open_remote_hdf5(url_for(name, month))
    try:
        latitudes = handle["latitude"][:]
        longitudes = handle["longitude"][:]
        rows, columns = _window(latitudes, longitudes, bounds)
        key = name.upper()
        values = handle[key][hour, rows.min():rows.max() + 1,
                             columns.min():columns.max() + 1]
        units = handle[key].attrs.get("units", b"").decode()
        selected_lat = latitudes[rows.min():rows.max() + 1]
        selected_lon = longitudes[columns.min():columns.max() + 1]
    finally:
        handle.close()

    values = np.asarray(values, dtype=float)
    if selected_lat[0] > selected_lat[-1]:
        selected_lat = selected_lat[::-1]
        values = values[::-1, :]
    return selected_lat, selected_lon, values, units, reader


def _window(latitudes, longitudes, bounds):
    (lat_low, lat_high), (lon_low, lon_high) = bounds
    rows = np.where((latitudes >= lat_low) & (latitudes <= lat_high))[0]
    columns = np.where((longitudes >= lon_low) & (longitudes <= lon_high))[0]
    if rows.size == 0 or columns.size == 0:
        raise ValueError("the window falls outside the grid")
    return rows, columns


def moisture_flux(month, bounds, hour=0):
    latitudes, longitudes, east, units, reader_east = load("viwve", month, bounds, hour)
    _, _, north, _, reader_north = load("viwvn", month, bounds, hour)
    field = VectorField(latitudes, longitudes, east, north, units=units)
    return field, (reader_east, reader_north)


def official_divergence(month, bounds, hour=0):
    return load("viwvd", month, bounds, hour)