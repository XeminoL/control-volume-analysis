import json
import http.server
import webbrowser
from pathlib import Path

import numpy as np

from analysis import (boundary_flux, circulation, interior_curl,
                      interior_divergence, summarise)
from field import from_hycom, polygon_area

HOST = "127.0.0.1"
PORT = 8765
DATA = "tonkin.nc"
PAGE = "index.html"
EDGE_SPACING = 2000.0
MIN_VERTICES = 3

_field = None


def field():
    global _field
    if _field is None:
        _field = from_hycom(DATA)
    return _field


def grid_payload():
    source = field()
    speed = np.hypot(source.east, source.north)
    finite = speed[np.isfinite(speed)]
    return {
        "latitudes": source.latitudes.tolist(),
        "longitudes": source.longitudes.tolist(),
        "east": _nullable(source.east),
        "north": _nullable(source.north),
        "speedMax": float(finite.max()) if finite.size else 0.0,
        "coverage": source.coverage(),
        "units": source.units,
    }


def _nullable(grid):
    return [[None if not np.isfinite(v) else round(float(v), 4) for v in row]
            for row in grid]


def analyse(vertices, with_uncertainty=False):
    source = field()
    budget = boundary_flux(vertices, source, spacing=EDGE_SPACING)
    inner, counted, skipped = interior_divergence(vertices, source)
    loop = circulation(vertices, source, spacing=EDGE_SPACING)
    curl, curl_cells = interior_curl(vertices, source)

    result = {
        "area": budget.area,
        "inflow": budget.inflow,
        "outflow": budget.outflow,
        "net": budget.net,
        "coverage": budget.coverage,
        "samples": budget.samples,
        "missing": budget.missing,
        "segments": budget.segments,
        "gauss": _pair(budget.net, inner),
        "green": _pair(loop, curl),
        "cells": counted,
        "skipped": skipped,
        "curlCells": curl_cells,
    }
    if with_uncertainty:
        result["uncertainty"] = _uncertainty(vertices, source)
    return result


def _pair(boundary, interior):
    gap = abs(boundary - interior)
    return {
        "boundary": boundary,
        "interior": interior,
        "gap": gap,
        "relative": gap / abs(interior) if interior else None,
    }


def _uncertainty(vertices, source):
    report = summarise(vertices, source)
    spread = report["spread"]
    low, high = spread.interval
    return {
        "nominal": spread.nominal,
        "deviation": spread.deviation,
        "relative": spread.relative,
        "low": low,
        "high": high,
        "signIsCertain": spread.sign_is_certain,
        "theoremGap": report["gap"]["absolute"],
        "theoremRelative": report["gap"]["against_gross"],
        "refinementChange": report["refinement_change"],
        "refinement": [[s, v] for s, v in report["refinement"]],
    }


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send_file(PAGE, "text/html; charset=utf-8")
        elif self.path == "/grid":
            self._send_json(grid_payload())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/analyse":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
            vertices = [(float(a), float(b)) for a, b in request["vertices"]]
        except (ValueError, KeyError, TypeError) as exc:
            self._send_json({"error": f"bad request: {exc}"}, status=400)
            return

        if len(vertices) < MIN_VERTICES:
            self._send_json({"error": "a boundary needs at least three points"},
                            status=400)
            return
        if abs(polygon_area(vertices)) == 0:
            self._send_json({"error": "the boundary encloses no area"},
                            status=400)
            return

        try:
            self._send_json(analyse(vertices, request.get("uncertainty", False)))
        except (ValueError, ZeroDivisionError) as exc:
            self._send_json({"error": str(exc)}, status=400)

    def _send_file(self, name, content_type):
        path = Path(__file__).parent / name
        if not path.exists():
            self.send_error(404, f"{name} is missing")
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, allow_nan=False, default=_plain).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


def _plain(value):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialise {type(value)}")


def main():
    source = field()
    print(f"{source}")
    print(f"serving http://{HOST}:{PORT}")
    server = http.server.ThreadingHTTPServer((HOST, PORT), Handler)
    webbrowser.open(f"http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()