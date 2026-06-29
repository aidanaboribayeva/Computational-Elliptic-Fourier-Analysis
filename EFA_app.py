from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
from numpy import (
    zeros, arange, pi, cos, sin, sqrt, arctan2, arctan, dot, roll, where,
    asarray, float64, newaxis, full_like, nan, abs, diff, unwrap, linspace,
    outer, sum as np_sum, mean, max as np_max, min as np_min, zeros_like
)
from scipy.special import elliprf, elliprd

# =============================================================================
# Paths + instruction images
# =============================================================================
APP_DIR = Path(__file__).resolve().parent
ASSETS_DIRS = [
    APP_DIR / "assets",
    APP_DIR / "Instruction images",
    APP_DIR,
]

import streamlit as st

def show_citation():
    """Display the citation block in the Streamlit app."""
    st.markdown("---")
    st.markdown("""
    **📚 If you use this app or code in your work, please cite:**

    **Boribayeva, A., Sultaniyar, S., Lukmanov, I., Baigarina, A., Rojas-Solórzano, L. R., Curtis, J. S., Govender N. & Golman, B. (2026). Integrated characterization, classification, and quasi-3D reconstruction of highly irregular particles using multiscale shape descriptors for predictive DEM flow simulation. *Powder Technology*, 122435.**
    """)


def _resolve_image(path_or_name: str) -> Optional[Path]:
    """Resolve an image by absolute path or by filename in ./assets or ./Instruction images."""
    try:
        p = Path(path_or_name)
        if p.exists():
            return p
    except Exception:
        pass

    name = Path(str(path_or_name)).name
    for base in ASSETS_DIRS:
        cand = base / name
        if cand.exists():
            return cand
    return None


def show_images(paths: List[str], title: str, expanded: bool, ncols: int = 2, image_width: int = 300) -> None:
    """Image grid inside an expander. Ignores missing images."""
    resolved: List[Path] = []
    for s in paths:
        p = _resolve_image(s)
        if p is not None:
            resolved.append(p)
    if not resolved:
        return

    with st.expander(title, expanded=expanded):
        cols = st.columns(ncols)
        for i, p in enumerate(resolved):
            with cols[i % ncols]:
                st.image(str(p), width=image_width)


# =============================================================================
# Data loader (CSV / TXT / TSV / Excel)
# =============================================================================

# -----------------------------------------------------------------------------
# Why each particle needs at least 3 points
# -----------------------------------------------------------------------------
# Every particle is described by an ordered list of (X, Y) points forming a
# CLOSED outline. The whole analysis (area, perimeter, Elliptic Fourier
# Descriptors, etc.) treats that outline as a polygon.
#   - 1 point  = just a dot           -> no shape, no area, no perimeter
#   - 2 points = a straight line      -> still no enclosed area
#   - 3 points = a triangle           -> the smallest real closed shape
# So 3 is the hard minimum below which the math is undefined. Particles with
# fewer points are skipped (not analyzed) rather than producing garbage numbers.
MIN_POINTS_PER_PARTICLE = 3

# File extensions the app knows how to read.
SUPPORTED_EXTENSIONS = ["csv", "txt", "tsv", "xlsx", "xls"]


class DataLoadError(ValueError):
    """Raised when an uploaded file cannot be read or does not contain valid ID/X/Y data.

    The message is written to be shown directly to the user, so it should explain
    both *what* went wrong and *what to do* about it.
    """
    pass


def _first_row_is_header(raw: pd.DataFrame) -> bool:
    """Decide whether the first row of a header-less read is actually a header.

    Rule: a header row contains text labels (e.g. "ID", "X", "Y"), so at least one
    value is non-numeric. A real data row (e.g. "0", "0", "0") is fully numeric.
    Therefore: first row is a header  <=>  it is NOT entirely numeric.
    """
    if raw is None or raw.empty:
        return False
    first = raw.iloc[0]
    numeric = pd.to_numeric(first, errors="coerce")
    return not bool(numeric.notna().all())


def _apply_header(raw: pd.DataFrame) -> pd.DataFrame:
    """Promote the first row to column names if it looks like a header.

    The file is always read with header=None first; here we fix it up:
      - header present  -> use row 0 as column names, keep the rest as data
      - no header        -> keep positional names (0, 1, 2, ...) and ALL rows as data
    This is what prevents a header-less file like "0,0,0 / 1,0,1 / ..." from losing
    its first row (which pandas would otherwise swallow as the header).
    """
    if _first_row_is_header(raw):
        header = raw.iloc[0].astype(str).str.strip().tolist()
        body = raw.iloc[1:].reset_index(drop=True)
        body.columns = header
        return body
    return raw


def _read_raw_table(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read an uploaded file into a raw DataFrame based on its extension.

    Supports CSV / TXT / TSV (delimiter is auto-detected) and Excel (.xlsx/.xls).
    Raises DataLoadError with a user-friendly message on any failure.
    """
    ext = Path(str(filename)).suffix.lower().lstrip(".")

    if ext not in SUPPORTED_EXTENSIONS:
        raise DataLoadError(
            f"Unsupported file type: '.{ext}'. "
            f"Please upload one of: {', '.join('.' + e for e in SUPPORTED_EXTENSIONS)}. "
            "The file must contain three columns: ID, X, Y."
        )

    # ---- Excel ----
    if ext in ("xlsx", "xls"):
        try:
            # Read WITHOUT assuming a header, then decide if the top row is one.
            raw = pd.read_excel(io.BytesIO(file_bytes), header=None)
        except Exception as e:
            raise DataLoadError(
                "Could not read the Excel file. Make sure it is a valid, "
                "non-empty .xlsx/.xls file with X, Y (and optionally ID) columns "
                f"on the first sheet. (Technical detail: {e})"
            )
        if raw is None or raw.empty:
            raise DataLoadError(
                "The Excel file was opened but is empty. "
                "It should contain three columns: ID, X, Y."
            )
        return _apply_header(raw)

    # ---- Text-based: CSV / TXT / TSV ----
    # Try common delimiters in order; the first that yields >= 3 columns wins.
    #   ","  ";"  "\t"  -> standard CSV/TSV style files (also handles "0, 0, 0" with
    #                      skipinitialspace=True, which strips the space after a comma)
    #   r"\s+"          -> space-separated OR column-aligned .txt (e.g. "A   0.0   0.0")
    #   None            -> last-resort auto-sniff (unreliable on aligned text, so it's last)
    # We read with header=None here and decide about the header separately in
    # _apply_header(), so that files WITHOUT a header row (just numbers) are not
    # corrupted by pandas treating their first data row as column names.
    raw = None
    for sep in [",", ";", "\t", r"\s+", None]:
        try:
            candidate = pd.read_csv(
                io.BytesIO(file_bytes),
                sep=sep,
                engine="python",
                header=None,
                skipinitialspace=True,  # turns "0, 0, 0" into clean 0 | 0 | 0
            )
            if candidate is not None and candidate.shape[1] >= 3:
                raw = candidate
                break
            if raw is None:
                raw = candidate  # keep something to report on if nothing reaches 3 cols
        except Exception:
            continue

    if raw is None or raw.empty:
        raise DataLoadError(
            f"Could not read the '.{ext}' file as a table. "
            "Make sure the columns are separated by commas, semicolons, tabs, or spaces, "
            "and that the file contains three columns: ID, X, Y."
        )
    return _apply_header(raw)


def _normalize_xy(df: pd.DataFrame) -> pd.DataFrame:
    """Pick the ID/X/Y columns, coerce to the right types, and drop bad rows."""
    df = df.rename(columns={c: str(c).strip() for c in df.columns})

    if df.shape[1] < 3:
        raise DataLoadError(
            f"The file has only {df.shape[1]} column(s), but 3 are required: ID, X, Y."
        )

    lower_map = {c.lower().strip(): c for c in df.columns}

    def pick(opts: List[str]) -> Optional[str]:
        for o in opts:
            if o in lower_map:
                return lower_map[o]
        return None

    id_col = pick(["id", "particle", "particle_id", "pid", "label"])
    x_col = pick(["x", "xcoord", "x_coord", "x-coordinate"])
    y_col = pick(["y", "ycoord", "y_coord", "y-coordinate"])

    if id_col is None or x_col is None or y_col is None:
        # Fall back to the first three columns by position.
        cols = list(df.columns[:3])
        id_col, x_col, y_col = cols[0], cols[1], cols[2]

    out = df[[id_col, x_col, y_col]].copy()
    out.columns = ["ID", "X", "Y"]
    out["ID"] = out["ID"].astype(str).str.strip()
    out["X"] = pd.to_numeric(out["X"], errors="coerce")
    out["Y"] = pd.to_numeric(out["Y"], errors="coerce")
    out = out.dropna(subset=["ID", "X", "Y"]).reset_index(drop=True)

    if out.empty:
        raise DataLoadError(
            "The file was read, but no valid numeric rows were found. "
            "Check that the X and Y columns contain numbers and that the file "
            "has ID, X, Y columns."
        )
    return out


def load_xy_table(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load any supported file into a clean ID/X/Y DataFrame.

    Raises DataLoadError (with a user-facing message) on any problem.
    """
    raw = _read_raw_table(file_bytes, filename)
    return _normalize_xy(raw)


# Backwards-compatible alias (older code / callers may still use this name).
def load_xy_csv(file_bytes: bytes) -> pd.DataFrame:
    return load_xy_table(file_bytes, "data.csv")


def particle_point_counts(df_xy: pd.DataFrame) -> pd.Series:
    """Number of coordinate points per particle ID."""
    return df_xy.groupby("ID").size()


def make_id_mapping(original_ids: List[str]) -> Dict[str, str]:
    return {oid: f"P{i:03d}" for i, oid in enumerate(original_ids, start=1)}


# =============================================================================
# Units (px to any unit)
# =============================================================================

# Define available units and their conversion factors (relative to mm)
UNIT_OPTIONS = {
    "micrometer (μm)": 0.001,
    "millimeter (mm)": 1.0,
    "centimeter (cm)": 10.0,
    "meter (m)": 1000.0,
    "inch (in)": 25.4,
}


def convert_px_to_unit(length_px: float, px_per_mm: float, target_unit: str) -> float:
    """Convert pixel length to target unit."""
    px_per_mm = float(max(px_per_mm, 1e-12))
    length_mm = length_px / px_per_mm
    conversion_factor = UNIT_OPTIONS.get(target_unit, 1.0)
    return length_mm / conversion_factor


def convert_px2_to_unit2(area_px2: float, px_per_mm: float, target_unit: str) -> float:
    """Convert pixel area to target unit squared."""
    px_per_mm = float(max(px_per_mm, 1e-12))
    area_mm2 = area_px2 / (px_per_mm ** 2)
    conversion_factor = UNIT_OPTIONS.get(target_unit, 1.0)
    return area_mm2 / (conversion_factor ** 2)


def get_unit_label(unit: str, is_area: bool = False) -> str:
    """Get formatted unit label for display."""
    if is_area:
        unit_name = unit.split()[0]
        if unit_name == "micrometer":
            return "μm²"
        elif unit_name == "millimeter":
            return "mm²"
        elif unit_name == "centimeter":
            return "cm²"
        elif unit_name == "meter":
            return "m²"
        elif unit_name == "inch":
            return "in²"
    else:
        unit_name = unit.split()[0]
        if unit_name == "micrometer":
            return "μm"
        elif unit_name == "millimeter":
            return "mm"
        elif unit_name == "centimeter":
            return "cm"
        elif unit_name == "meter":
            return "m"
        elif unit_name == "inch":
            return "in"
    return unit


# =============================================================================
# Geometry helpers
# =============================================================================
def _ensure_closed(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) >= 3 and (x[0] != x[-1] or y[0] != y[-1]):
        x = np.append(x, x[0])
        y = np.append(y, y[0])
    return x, y


def OutlineArea(x, y):
    x = asarray(x, dtype=float)
    y = asarray(y, dtype=float)
    x, y = _ensure_closed(x, y)
    y_rolled = roll(y, -1)
    x_rolled = roll(x, -1)
    term1 = x * y_rolled
    term2 = x_rolled * y
    cross_sum = np_sum(term1 - term2)
    area = abs(cross_sum) / 2.0
    return area


def OutlineCircumference(x, y):
    x = asarray(x)
    y = asarray(y)
    x, y = _ensure_closed(x, y)
    x_rolled = roll(x, -1)
    y_rolled = roll(y, -1)
    dx = x_rolled - x
    dy = y_rolled - y
    SegmentLengths = sqrt(dx ** 2 + dy ** 2)
    PolygonLength = np_sum(SegmentLengths)
    return PolygonLength


def OutlineCentroid(x, y):
    NoP = len(x)
    xmid = np_sum(x) / float(NoP)
    ymid = np_sum(y) / float(NoP)
    return (xmid, ymid)


def ELLE(ak):
    pi2 = pi / 2.
    s = 1.
    cc = 0.
    Q = (1. - s * ak) * (1. + s * ak)
    ELLE_val = s * (elliprf(cc, Q, 1.) - ((s * ak) * (s * ak)) * elliprd(cc, Q, 1.) / 3.)
    return ELLE_val


def ComputeEllFourierCoef(No_OutlinePoints, X, Y, No_harmonics):
    X = asarray(X, dtype=float64)
    Y = asarray(Y, dtype=float64)
    X_shifted = roll(X, -1)
    Y_shifted = roll(Y, -1)
    DXi = X_shifted - X
    DYi = Y_shifted - Y
    DTi = sqrt(DXi ** 2 + DYi ** 2)
    DTi = where(DTi == 0, 1e-12, DTi)
    R_x = DXi / DTi
    R_y = DYi / DTi
    T_curr = np.cumsum(DTi)
    T_prev = np.concatenate(([0.0], T_curr[:-1]))
    PolygonLength = T_curr[-1]

    Tsum = 0.0
    Xsum = 0.0
    Ysum = 0.0
    Ax0_integral_sum = 0.0
    Ay0_integral_sum = 0.0

    for i in range(No_OutlinePoints):
        DT_i = DTi[i]
        Tnew = Tsum + DT_i
        Rx_i = R_x[i]
        Ry_i = R_y[i]
        Xi_i = Xsum - Rx_i * Tsum
        Delta_i = Ysum - Ry_i * Tsum
        Ax0_integral_sum += 0.5 * Rx_i * (Tnew * Tnew - Tsum * Tsum) + Xi_i * DT_i
        Ay0_integral_sum += 0.5 * Ry_i * (Tnew * Tnew - Tsum * Tsum) + Delta_i * DT_i
        Tsum = Tnew
        Xsum += DXi[i]
        Ysum += DYi[i]

    if PolygonLength > 0.0:
        Ax0 = X[0] + Ax0_integral_sum / PolygonLength
        Ay0 = Y[0] + Ay0_integral_sum / PolygonLength
    else:
        Ax0 = X[0]
        Ay0 = Y[0]

    Ax = zeros(No_harmonics + 1)
    Bx = zeros(No_harmonics + 1)
    Ay = zeros(No_harmonics + 1)
    By = zeros(No_harmonics + 1)

    Ax[0] = Ax0
    Ay[0] = Ay0

    if No_harmonics == 0 or PolygonLength == 0.0:
        return (Ax, Bx, Ay, By)

    j_harmonics = arange(1, No_harmonics + 1, dtype=np.float64)
    c1_j = 2.0 * pi * j_harmonics / PolygonLength
    Theta_curr = outer(T_curr, c1_j)
    Theta_prev = outer(T_prev, c1_j)
    DiffCos = cos(Theta_curr) - cos(Theta_prev)
    DiffSin = sin(Theta_curr) - sin(Theta_prev)

    AxSUM = dot(R_x, DiffCos)
    BxSUM = dot(R_x, DiffSin)
    AySUM = dot(R_y, DiffCos)
    BySUM = dot(R_y, DiffSin)

    c2_j = PolygonLength / (2.0 * pi ** 2 * j_harmonics ** 2)

    Ax[1:] = AxSUM * c2_j
    Bx[1:] = BxSUM * c2_j
    Ay[1:] = AySUM * c2_j
    By[1:] = BySUM * c2_j

    return (Ax, Bx, Ay, By)


def InverseEllFourier(Ax, Bx, Ay, By, No_XY):
    No_harmonics = len(Ax) - 1
    n_harmonics = arange(1, No_harmonics + 1)
    i_points = arange(No_XY)
    M = outer(i_points, n_harmonics) * (2.0 * pi / No_XY)
    cos_M = cos(M)
    sin_M = sin(M)
    Ax_n = Ax[1:]
    Bx_n = Bx[1:]
    Ay_n = Ay[1:]
    By_n = By[1:]
    Xest = Ax[0] + dot(cos_M, Ax_n) + dot(sin_M, Bx_n)
    Yest = Ay[0] + dot(cos_M, Ay_n) + dot(sin_M, By_n)
    return (Xest, Yest)


def FourierCoefNormalization(Ax, Bx, Ay, By, No_harmonics, FlagLocation, FlagScale, FlagRotation, FlagStart):
    Ax0 = Ax[0]
    Ay0 = Ay[0]
    Ax1 = Ax[1]
    Bx1 = Bx[1]
    Ay1 = Ay[1]
    By1 = By[1]

    Denom = Ax1 * Ax1 + Ay1 * Ay1 - Bx1 * Bx1 - By1 * By1
    if Denom == 0:
        Denom = 1e-12

    theta = 0.5 * arctan2(2. * (Ax1 * Bx1 + Ay1 * By1), Denom)
    Astar = Ax1 * cos(theta) + Bx1 * sin(theta)
    Cstar = Ay1 * cos(theta) + By1 * sin(theta)
    psi = arctan2(Cstar, Astar)
    Estar = sqrt(Astar * Astar + Cstar * Cstar)

    if Estar == 0.:
        Estar = 1.
    if FlagScale == 0:
        Estar = 1.
    if FlagRotation == 0:
        psi = 0.
    if FlagStart == 0:
        theta = 0.

    cos_psi = cos(psi)
    sin_psi = sin(psi)

    Ax0N = (cos_psi * Ax0 + sin_psi * Ay0) / Estar
    Ay0N = (-sin_psi * Ax0 + cos_psi * Ay0) / Estar

    for i in range(1, No_harmonics + 1):
        cos_theta = cos(float(i) * theta)
        sin_theta = sin(float(i) * theta)
        c1 = Ax[i] * cos_psi + Ay[i] * sin_psi
        c2 = Bx[i] * cos_psi + By[i] * sin_psi
        c3 = Ay[i] * cos_psi - Ax[i] * sin_psi
        c4 = By[i] * cos_psi - Bx[i] * sin_psi
        Ax[i] = (c1 * cos_theta + c2 * sin_theta) / Estar
        Bx[i] = (c2 * cos_theta - c1 * sin_theta) / Estar
        Ay[i] = (c3 * cos_theta + c4 * sin_theta) / Estar
        By[i] = (c4 * cos_theta - c3 * sin_theta) / Estar

    Scale1 = Estar
    RotateAngle = psi
    StartAngle = theta
    Ax[0] = Ax0N
    Ay[0] = Ay0N
    Bx[0] = 0.
    By[0] = 0.

    return (Ax, Bx, Ay, By, Scale1, RotateAngle, StartAngle)


def CoefNormalization(X, Y, XMID, YMID, FlagLocation, FlagScale, FlagRotation, FlagStart, No_harmonics=16):
    NoP = len(X)
    if FlagLocation == 1:
        X1 = zeros(len(X))
        Y1 = zeros(len(Y))
        X1 = X - XMID
        Y1 = Y - YMID
        Ax, Bx, Ay, By = ComputeEllFourierCoef(NoP, X1, Y1, No_harmonics)
    else:
        Ax, Bx, Ay, By = ComputeEllFourierCoef(NoP, X, Y, No_harmonics)

    Ax, Bx, Ay, By, Scale1, RotateAngle, StartAngle = FourierCoefNormalization(
        Ax, Bx, Ay, By, No_harmonics, FlagLocation, FlagScale, FlagRotation, FlagStart
    )
    return (Ax, Bx, Ay, By, Scale1, RotateAngle, StartAngle)


def ShapeIndex(Ax, Bx, Ay, By, FlagRotation, psi, XMID, YMID, Scale1, X, Y, N0_sum, LargeNumber):
    Bk1 = zeros(64)
    Ask = zeros(64)
    Bsk = zeros(64)
    Bk2 = zeros(N0_sum + 1)

    Ax1 = Ax[1]
    Bx1 = Bx[1]
    Ay1 = Ay[1]
    By1 = By[1]
    cos_psi, sin_psi = cos(psi), sin(psi)

    if FlagRotation == 0:
        denom = Ax1 ** 2 + Ay1 ** 2 - Bx1 ** 2 - By1 ** 2
        if denom == 0:
            denom = 1e-12
        theta = 0.5 * arctan2(2. * (Ax1 * Bx1 + Ay1 * By1), denom)
        Astar = Ax1 * cos(theta) + Bx1 * sin(theta)
        Cstar = Ay1 * cos(theta) + By1 * sin(theta)
        psi = arctan2(Cstar, Astar)
        cos_psi, sin_psi = cos(psi), sin(psi)
        c1 = Ax1 * cos_psi + Ay1 * sin_psi
        c2 = Bx1 * cos_psi + By1 * sin_psi
        c3 = Ay1 * cos_psi - Ax1 * sin_psi
        c4 = By1 * cos_psi - Bx1 * sin_psi
        Ax1 = c1
        Bx1 = c2
        Ay1 = c3
        By1 = c4

    det = abs(Bx1 * Ay1 - Ax1 * By1)
    denom_ae = sqrt(Ay1 ** 2 + By1 ** 2)
    denom_be = sqrt(Ax1 ** 2 + Bx1 ** 2)
    ae = det / denom_ae if denom_ae != 0 else nan
    be = det / denom_be if denom_be != 0 else nan
    k = be / ae if ae != 0 and not np.isnan(ae) else nan

    No_OutlinePoints = len(X)

    X_shifted = X - XMID
    Y_shifted = Y - YMID
    xp = (X_shifted * cos_psi + Y_shifted * sin_psi) / Scale1
    yp = (-X_shifted * sin_psi + Y_shifted * cos_psi) / Scale1

    rd = sqrt(xp ** 2 + yp ** 2)
    th = arctan2(yp, xp)

    if ae is not None and not np.isnan(ae) and be is not None and not np.isnan(be):
        ae_sq, be_sq = ae ** 2, be ** 2
        cos_th_sq = cos(th) ** 2
        re_denom = sqrt(ae_sq - (ae_sq - be_sq) * cos_th_sq)
        re = np.divide(ae * be, re_denom, out=full_like(re_denom, nan), where=re_denom != 0)
        Sr = np_sum((rd - re) ** 2) / float(No_OutlinePoints)

        xk = 1. - (be_sq / ae_sq)
        xk = np.clip(xk, 0, 1)
        e2 = ELLE(sqrt(xk))
        Le = 4. * ae * e2
        rc = Le / (2. * pi)
        Uc = sqrt(Sr) / rc if rc != 0 and not np.isnan(rc) else nan
    else:
        Sr, Le, rc, Uc = nan, nan, nan, nan

    xp1 = roll(xp, -1)
    yp1 = roll(yp, -1)
    dx = xp1 - xp
    dy = yp1 - yp

    with np.errstate(divide='ignore', invalid='ignore'):
        di = where(np.abs(dx) <= 1E-30, 1000.0, dy / dx)
        di = where(np.isfinite(di), di, 1000.0)

    jj_range = arange(36)
    theta_jj = (2. * pi / 36.) * jj_range
    cos_jj, sin_jj = cos(theta_jj), sin(theta_jj)
    di_expanded = di[:, newaxis]
    numerator = di_expanded * cos_jj - sin_jj
    denominator = cos_jj + di_expanded * sin_jj
    ratio = np.divide(numerator, denominator, out=zeros_like(numerator), where=denominator != 0)
    aver_sum = np_sum(ratio, axis=1)
    aver = arctan(aver_sum / 36.)

    j_range = arange(1, 65)
    i_plus_1 = arange(1, No_OutlinePoints + 1)
    arg = (2. * pi / No_OutlinePoints) * i_plus_1[:, newaxis] * j_range[newaxis, :]
    cos_arg, sin_arg = cos(arg), sin(arg)
    aver_expanded = aver[:, newaxis]
    say_i = aver_expanded * cos_arg
    sby_i = aver_expanded * sin_arg
    say_vec = np_sum(say_i, axis=0) / float(No_OutlinePoints)
    sby_vec = np_sum(sby_i, axis=0) / float(No_OutlinePoints)
    sint1 = sqrt(say_vec ** 2 + sby_vec ** 2)
    SUM_Bk1 = np_sum(sint1)
    S1 = 1.0 / SUM_Bk1 if SUM_Bk1 != 0 else nan
    Bk1[:] = sint1
    Ask[:64] = say_vec
    Bsk[:64] = sby_vec

    j_range = arange(1, N0_sum + 1)
    arg = (2. * pi / No_OutlinePoints) * i_plus_1[:, newaxis] * j_range[newaxis, :]
    cos_arg, sin_arg = cos(arg), sin(arg)
    say_i = aver_expanded * cos_arg
    sby_i = aver_expanded * sin_arg
    say_vec = np_sum(say_i, axis=0) / float(No_OutlinePoints)
    sby_vec = np_sum(sby_i, axis=0) / float(No_OutlinePoints)
    say1 = say_vec[0] if len(say_vec) > 0 else 1.0
    sby1 = sby_vec[0] if len(sby_vec) > 0 else 1.0

    SUM_Bk2 = sqrt(2.0)
    Bk2[1] = sqrt(2.0)

    if len(say_vec) > 1 and len(sby_vec) > 1 and say1 != 0 and sby1 != 0:
        sint2_slice = sqrt((say_vec[1:] / say1) ** 2 + (sby_vec[1:] / sby1) ** 2)
        sint2_norm = where(sint2_slice > LargeNumber, 0.0, sint2_slice)
        SUM_Bk2 += np_sum(sint2_norm)
        Bk2[2:N0_sum + 1] = sint2_norm[:min(len(sint2_norm), N0_sum - 1)]
    else:
        sint2_norm = zeros(min(N0_sum - 1, max(0, len(say_vec) - 1)))

    S2 = 1.0 / SUM_Bk2 if SUM_Bk2 != 0 else nan

    return (ae, be, k, Le, rc, Uc, S1, S2, Bk1, Bk2, Ask, Bsk)


def ShapeIndicesEF(Ax, Bx, Ay, By, No_sum, LargeLimit, SmallLimit, XMID, YMID, FlagScale,
                   Scale1, RotateAngle, StartAngle, No_OutlinePoints, X, Y, No_harmonics):
    FlagRotation = 1
    psi = RotateAngle

    ae, be, k, Le, rc, Uc, S1, S2, Bk1, Bk2, Ask, Bsk = ShapeIndex(
        Ax, Bx, Ay, By, FlagRotation, psi, XMID, YMID, Scale1, X, Y, No_sum, LargeLimit
    )

    Ax, Bx, Ay, By = ComputeEllFourierCoef(No_OutlinePoints, X, Y, No_harmonics)

    NoAss = min(No_harmonics, No_sum)
    Ax_slice = Ax[1:NoAss + 1]
    Bx_slice = Bx[1:NoAss + 1]
    Ay_slice = Ay[1:NoAss + 1]
    By_slice = By[1:NoAss + 1]

    abs_Ax = abs(Ax_slice)
    abs_Bx = abs(Bx_slice)
    abs_Ay = abs(Ay_slice)
    abs_By = abs(By_slice)

    condition_Bx = (abs_Bx > SmallLimit) & (abs_Ax > SmallLimit)
    ratio_Bx_Ax = np.divide(abs_Bx, abs_Ax, out=np.zeros_like(abs_Bx), where=condition_Bx)
    SumBx = np.sum(ratio_Bx_Ax)

    condition_Ay = (abs_Ay > SmallLimit) & (abs_By > SmallLimit)
    ratio_Ay_By = np.divide(abs_Ay, abs_By, out=np.zeros_like(abs_Ay), where=condition_Ay)
    SumAy = np.sum(ratio_Ay_By)

    Ass2 = np.sqrt(SumBx * SumAy) if (SumBx * SumAy) > 0 else 0.0

    if No_harmonics >= 2:
        NoAss = min(15, No_harmonics)
        Ax_slice = Ax[2:NoAss + 1]
        Bx_slice = Bx[2:NoAss + 1]
        Ay_slice = Ay[2:NoAss + 1]
        By_slice = By[2:NoAss + 1]

        abs_Ax = abs(Ax_slice)
        abs_Ay = abs(Ay_slice)
        abs_Bx = abs(Bx_slice)
        abs_By = abs(By_slice)

        SumAx = np.sum(np.where(abs_Ax > SmallLimit, abs_Ax, 0.0))
        SumAy = np.sum(np.where(abs_Ay > SmallLimit, abs_Ay, 0.0))
        SumBx = np.sum(np.where(abs_Bx > SmallLimit, abs_Bx, 0.0))
        SumBy = np.sum(np.where(abs_By > SmallLimit, abs_By, 0.0))

        Ass1 = 0.0
        if SumAx > 0.0 and SumBy > 0.0:
            Ass1 = np.sqrt((SumBx / SumAx) * (SumAy / SumBy))

        NoMaxAx1 = 1
        NoMaxBy1 = 1
        MaxAx = abs(Ax[1])
        MaxBy = abs(By[1])

        j = 1
        while j <= No_harmonics:
            if MaxAx < abs(Ax[j]):
                MaxAx = abs(Ax[j])
                NoMaxAx1 = j
            if MaxBy < abs(By[j]):
                MaxBy = abs(By[j])
                NoMaxBy1 = j
            j = j + 1

        j = 1
        MaxAx2 = -1
        MaxBy2 = -1
        NoMaxAx = 1
        NoMaxBy = 1

        while j <= No_harmonics:
            if j != NoMaxAx1:
                if MaxAx2 < abs(Ax[j]):
                    MaxAx2 = abs(Ax[j])
                    NoMaxAx = j
            if j != NoMaxBy1:
                if MaxBy2 < abs(By[j]):
                    MaxBy2 = abs(By[j])
                    NoMaxBy = j
            j = j + 1

        Px = NoMaxAx + 1
        Py = NoMaxBy + 1
        P = np.sqrt(Px * Py)
    else:
        Ass1 = 0.0
        P = 1.0

    return (ae, be, k, Le, rc, Uc, S1, S2, Bk1, Bk2, Ask, Bsk, Ass1, Ass2, P)


def ComputeAngularity(Ax, Bx, Ay, By, No_harmonics, w=360):
    if No_harmonics < 1 or len(Ax) < 2:
        return 0.0

    TwoPi = 2.0 * pi
    angles = np.linspace(0, TwoPi, w, endpoint=False)
    n_array = np.arange(1, No_harmonics + 1)

    n_u = np.outer(angles, n_array)
    sin_n_u = np.sin(n_u)
    cos_n_u = np.cos(n_u)

    Ax_n = Ax[1:No_harmonics + 1]
    Bx_n = Bx[1:No_harmonics + 1]
    Ay_n = Ay[1:No_harmonics + 1]
    By_n = By[1:No_harmonics + 1]

    nAx = n_array * Ax_n
    nBx = n_array * Bx_n
    nAy = n_array * Ay_n
    nBy = n_array * By_n

    x_deriv = np.sum(-nAx[:, np.newaxis] * sin_n_u.T + nBx[:, np.newaxis] * cos_n_u.T, axis=0)
    y_deriv = np.sum(-nAy[:, np.newaxis] * sin_n_u.T + nBy[:, np.newaxis] * cos_n_u.T, axis=0)

    hx = np.arctan2(y_deriv, x_deriv)
    hx = np.unwrap(hx)
    dh = np.abs(np.diff(hx, append=hx[0]))
    AIg = (1.0 / TwoPi) * np.sum(dh) - 1.0
    return AIg


def ShapeFactors(NoP, X, Y, Area, Perimeter):
    Pi = np.pi
    Xm = np.mean(X)
    Ym = np.mean(Y)
    N_angle = 180
    delta_psi = Pi / (N_angle - 1)
    psi = 0.0
    HeightMin = 1e10
    LengthMax = 0.0
    AngleMax = 0.0

    for j in range(N_angle):
        Xwork = (X - Xm) * np.cos(psi) + (Y - Ym) * np.sin(psi)
        Ywork = -(X - Xm) * np.sin(psi) + (Y - Ym) * np.cos(psi)
        Xmin = np.min(Xwork)
        Xmax = np.max(Xwork)
        Ymin = np.min(Ywork)
        Ymax = np.max(Ywork)
        LengthCurrent = abs(Xmin) + abs(Xmax)
        HeightCurrent = abs(Ymin) + abs(Ymax)
        if HeightCurrent < HeightMin:
            HeightMin = HeightCurrent
            LengthMax = LengthCurrent
            AngleMax = psi
        psi += delta_psi

    Elongation = LengthMax / HeightMin if HeightMin > 0 else nan
    Bulkiness = (LengthMax * HeightMin) / Area if Area > 0 else nan
    Surface = (Perimeter ** 2) / (4 * Pi * Area) if Area > 0 else nan
    Circularity = 2.0 * np.sqrt(Pi * Area) / Perimeter if Perimeter > 0 else nan

    Xshift = X - Xm
    Yshift = Y - Ym
    distances = np.sqrt(Xshift ** 2 + Yshift ** 2)
    radius = np.max(distances)

    if radius <= 0 or Area <= 0:
        Sphericity = 0.0
    else:
        Sphericity = np.sqrt(Area / Pi) / radius

    return Elongation, Bulkiness, Surface, Circularity, Sphericity


# =============================================================================
# Fixed computation params
# =============================================================================
@dataclass(frozen=True)
class ShapeParams:
    No_harmonics: int = 16
    no_sum: int = 16
    large_limit: float = 100.0
    small_limit: float = 1e-5
    flag_location: int = 1
    flag_scale: int = 1
    flag_rotation: int = 1
    flag_start: int = 1


PARAMS_FIXED = ShapeParams(
    No_harmonics=16,
    no_sum=16,
    large_limit=100.0,
    small_limit=1e-5,
    flag_location=1,
    flag_scale=1,
    flag_rotation=1,
    flag_start=1,
)


# =============================================================================
# Batch computation
# =============================================================================
def compute_shape_indices(df_xy: pd.DataFrame, params: ShapeParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    results = []
    errors = []

    for pid, g in df_xy.groupby("ID"):
        try:
            X = g["X"].to_numpy(dtype=float)
            Y = g["Y"].to_numpy(dtype=float)
            if len(X) < 3:
                raise ValueError("Too few points")

            X, Y = _ensure_closed(X, Y)
            NoP = len(X)

            XMID, YMID = OutlineCentroid(X, Y)

            Ax, Bx, Ay, By, Scale1, RotateAngle, StartAngle = CoefNormalization(
                X, Y, XMID, YMID,
                params.flag_location,
                params.flag_scale,
                params.flag_rotation,
                params.flag_start,
                params.No_harmonics
            )

            ae, be, k, Le, rc, Uc, S1, S2, Bk1, Bk2, Ask, Bsk, Ass1, Ass2, P = ShapeIndicesEF(
                Ax, Bx, Ay, By,
                params.no_sum,
                params.large_limit,
                params.small_limit,
                XMID, YMID,
                params.flag_scale,
                Scale1,
                RotateAngle,
                StartAngle,
                NoP,
                X,
                Y,
                params.No_harmonics
            )

            Ax_raw, Bx_raw, Ay_raw, By_raw = ComputeEllFourierCoef(NoP, X, Y, params.No_harmonics)
            AIg = ComputeAngularity(Ax_raw, Bx_raw, Ay_raw, By_raw, No_harmonics=params.no_sum, w=360)

            area = OutlineArea(X, Y)
            perimeter = OutlineCircumference(X, Y)

            elongation, bulkiness, surface, circularity, sphericity = ShapeFactors(
                NoP, X, Y, area, perimeter
            )

            results.append({
                'ID': pid,
                'Kael': k,
                'angularity': AIg,
                'surface_roughness': Uc,
                'asymmetricity1': Ass1,
                'asymmetricity2': Ass2,
                'polygonality': P,
                'area': area,
                'perimeter': perimeter,
                'elongation': elongation,
                'bulkiness': bulkiness,
                'surface': surface,
                'circularity': circularity,
                'sphericity': sphericity
            })

        except Exception as e:
            errors.append({"ID": pid, "error": str(e)})
            print(f"Error processing particle {pid}: {str(e)}")

    return pd.DataFrame(results), pd.DataFrame(errors)


# =============================================================================
# Module 1 plots
# =============================================================================
def plot_reconstruction(x: np.ndarray, y: np.ndarray, K: int):
    x, y = _ensure_closed(x, y)
    NoP = len(x)

    Ax, Bx, Ay, By = ComputeEllFourierCoef(NoP, x, y, K)
    xt, yt = InverseEllFourier(Ax, Bx, Ay, By, NoP)

    fig, ax = plt.subplots(figsize=(3.6, 3.6), dpi=600)
    ax.plot(x, y, 'b-', linewidth=1.3, label="Original outline")
    ax.plot(xt, yt, 'r--', linewidth=1.3, label=f"Reconstructed (K={K})")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Outline reconstruction", fontsize=11)
    return fig, xt, yt


def fit_ellipse_equal_area(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cx, cy = OutlineCentroid(x, y)
    X = np.stack([x - cx, y - cy], axis=1)

    cov = np.cov(X.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    area = OutlineArea(x, y)
    if area <= 0:
        area = float(np.pi * max(vals[0], 1e-12) * max(vals[1], 1e-12))

    a0 = math.sqrt(max(vals[0], 1e-12))
    b0 = math.sqrt(max(vals[1], 1e-12))
    scale = math.sqrt(area / (math.pi * a0 * b0 + 1e-12))

    a = a0 * scale
    b = b0 * scale
    theta = math.atan2(vecs[1, 0], vecs[0, 0])

    if b > a:
        a, b = b, a
        theta += math.pi / 2.0

    return cx, cy, a, b, theta


def equal_area_circle_radius(x: np.ndarray, y: np.ndarray) -> float:
    area = OutlineArea(x, y)
    return float(math.sqrt(area / math.pi)) if area > 0 else 0.0


def plot_ellipse_and_circle(x: np.ndarray, y: np.ndarray):
    x, y = _ensure_closed(x, y)

    cx, cy, a, b, theta = fit_ellipse_equal_area(x, y)
    r = equal_area_circle_radius(x, y)

    t = np.linspace(0, 2 * np.pi, 300)
    ex = cx + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
    ey = cy + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)

    cxr = cx + r * np.cos(t)
    cyr = cy + r * np.sin(t)

    fig, ax = plt.subplots(figsize=(3.6, 3.6), dpi=170)
    ax.plot(x, y, 'b-', linewidth=1.3, label="Original outline")
    ax.plot(ex, ey, 'g-', linewidth=1.3, label="Ellipse")
    ax.plot(cxr, cyr, 'm-', linewidth=1.3, label="Circle")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Equal-area ellipse and circle", fontsize=11)

    return fig


# =============================================================================
# Contour-fit metrics
# =============================================================================
def _resample_closed_contour(x: np.ndarray, y: np.ndarray, m: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    x, y = _ensure_closed(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if len(x) < 4:
        return x, y

    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.hypot(dx, dy)
    ds[ds == 0] = 1e-12
    s = np.concatenate(([0.0], np.cumsum(ds)))
    total = float(s[-1]) if float(s[-1]) > 0 else 1.0

    u = np.linspace(0.0, total, m, endpoint=False)
    xr = np.interp(u, s, x)
    yr = np.interp(u, s, y)
    return xr, yr


def _best_shift_sse(A: np.ndarray, B: np.ndarray) -> float:
    m = A.shape[0]
    best = np.inf
    for k in range(m):
        sse = float(np.sum((A - np.roll(B, k, axis=0)) ** 2))
        if sse < best:
            best = sse
    return best


def contour_fit_metrics(x, y, xt, yt, m: int = 200) -> Tuple[float, float]:
    x1, y1 = _resample_closed_contour(x, y, m=m)
    x2, y2 = _resample_closed_contour(xt, yt, m=m)

    A = np.stack([x1, y1], axis=1)
    B = np.stack([x2, y2], axis=1)

    sse = min(_best_shift_sse(A, B), _best_shift_sse(A, B[::-1]))
    sst = float(np.sum((A - A.mean(axis=0)) ** 2))

    r2 = 1.0 - sse / max(sst, 1e-12)
    nrmse = float(math.sqrt(sse / max(sst, 1e-12)))
    return r2, nrmse


# =============================================================================
# Streamlit UI
# =============================================================================
st.set_page_config(page_title="Shape Analysis", layout="wide")
st.title("🔬 Shape Analysis with Elliptic Fourier Descriptors")

uploaded = st.sidebar.file_uploader(
    "Upload data file (ID, X, Y)",
    type=SUPPORTED_EXTENSIONS,
    help=(
        "Supported formats: "
        + ", ".join("." + e for e in SUPPORTED_EXTENSIONS)
        + ". The file must contain three columns: ID, X, Y."
    ),
)
# Visible note (not just a tooltip) so users immediately see which formats work.
st.sidebar.caption(
    "✅ Accepted formats: "
    + ", ".join("**." + e + "**" for e in SUPPORTED_EXTENSIONS)
    + ". Columns can be separated by commas, tabs, or spaces."
)

EXPANDED = True

st.sidebar.markdown("---")
# IMPORTANT: module selection must be defined BEFORE the if uploaded is None check
module = st.sidebar.radio("Modules", ["Module 1", "Module 2", "Module 3"], index=0)

st.sidebar.markdown("---")
with st.sidebar.expander("📏 Set Scale & Units", expanded=True):
    px_per_mm = st.number_input("px/mm calibration", min_value=0.0001, value=77.0, step=0.1,
                                help="Number of pixels per millimeter in your image")

    selected_unit = st.selectbox(
        "Output unit",
        options=list(UNIT_OPTIONS.keys()),
        index=1,
        help="Select the unit for area and perimeter measurements"
    )

    st.caption(
        f"Example: 1 px = {1 / px_per_mm:.4f} mm = {1 / px_per_mm / UNIT_OPTIONS[selected_unit]:.4f} {selected_unit.split()[0]}")

if uploaded is None:
    st.info("⬅️ Upload a data file to start.")
    st.caption(
        "Supported formats: "
        + ", ".join("." + e for e in SUPPORTED_EXTENSIONS)
        + ". Expected columns: ID, X, Y."
    )
    st.stop()

file_bytes = uploaded.getvalue()
file_hash = hashlib.md5(file_bytes).hexdigest()


@st.cache_data(show_spinner=False)
def _load_cached(h: str, b: bytes, name: str) -> pd.DataFrame:
    return load_xy_table(b, name)


try:
    df_xy = _load_cached(file_hash, file_bytes, uploaded.name)
except DataLoadError as e:
    # Friendly, actionable message for an unreadable / wrong-format / empty file.
    st.error(f"❌ Could not use this file.\n\n{e}")
    st.stop()
except Exception as e:
    st.error(
        "❌ Unexpected error while reading the file. Please check that it is a valid "
        f"data file with ID, X, Y columns.\n\n(Technical detail: {e})"
    )
    st.stop()

if df_xy.empty:
    st.error(
        "❌ The file was read but contains no valid rows. "
        "Make sure the ID, X, and Y columns are present and that X/Y are numbers."
    )
    st.stop()

# -----------------------------------------------------------------------------
# Point-count check (the "fewer than 3 points" model)
# -----------------------------------------------------------------------------
# We count how many (X, Y) points each particle has, then split them into:
#     valid     -> count >= MIN_POINTS_PER_PARTICLE  (can be analyzed)
#     too_small -> count <  MIN_POINTS_PER_PARTICLE  (must be skipped)
# Two outcomes:
#   1) If NOTHING is valid  -> hard stop with a clear explanation.
#   2) If SOME are valid     -> keep going, but warn which ones are skipped and why.
counts = particle_point_counts(df_xy)                      # points per particle ID
valid_particles = counts[counts >= MIN_POINTS_PER_PARTICLE]
too_small = counts[counts < MIN_POINTS_PER_PARTICLE]

# Case 1: no usable particle at all -> stop and explain.
if valid_particles.empty:
    largest = int(counts.max()) if len(counts) else 0

    # Special, very common cause: the FIRST column is a point index (0,1,2,3...)
    # rather than a particle ID, so every row becomes its own 1-point "particle".
    every_row_is_a_particle = (len(counts) == len(df_xy)) and bool((counts == 1).all())
    hint = ""
    if every_row_is_a_particle:
        hint = (
            "\n\n💡 **Most likely cause:** the first column is being read as the "
            "particle **ID**, but here every row has a different value in it "
            "(e.g. 0, 1, 2, 3...), so each point is treated as a separate particle. "
            "If all these points belong to **one** particle, give them the **same ID** "
            "in the first column, like:\n\n"
            "```\nID,X,Y\n0,0,0\n0,0,1\n0,1,1\n0,1,0\n```\n"
            "(Same ID `0` on every row = one particle with 4 points.)"
        )

    st.error(
        f"❌ Nothing can be analyzed: every particle has fewer than "
        f"**{MIN_POINTS_PER_PARTICLE}** points.\n\n"
        f"A particle outline is a closed shape, so it needs at least "
        f"**{MIN_POINTS_PER_PARTICLE}** (X, Y) points (a triangle is the smallest "
        f"possible shape). The biggest particle in this file has only **{largest}** "
        f"point(s)."
        + hint
        + f"\n\n**What to do:** make sure each particle's outline has at least "
        f"{MIN_POINTS_PER_PARTICLE} points sharing the same ID, and that X and Y are "
        f"numbers (text values get dropped automatically)."
    )
    st.stop()

# Case 2: some particles are too small -> warn, list them, and show the counts.
if not too_small.empty:
    preview = ", ".join(str(i) for i in too_small.index[:10])
    more = "" if len(too_small) <= 10 else f" (+{len(too_small) - 10} more)"
    st.warning(
        f"⚠️ {len(too_small)} of {len(counts)} particle(s) have fewer than "
        f"{MIN_POINTS_PER_PARTICLE} points and will be **skipped**: {preview}{more}.\n\n"
        f"The remaining **{len(valid_particles)}** particle(s) will still be analyzed."
    )
    # Small table so the user can see exactly which IDs and how many points each has.
    with st.expander("See skipped particles"):
        skipped_tbl = (
            too_small.rename("Points")
            .reset_index()
            .rename(columns={"index": "ID"})
            .sort_values("Points")
        )
        st.dataframe(skipped_tbl, use_container_width=True, hide_index=True)

original_ids = df_xy["ID"].dropna().astype(str).unique().tolist()
id_map = make_id_mapping(original_ids)
reverse_map = {v: k for k, v in id_map.items()}

st.caption(f"Detected **{df_xy['ID'].nunique()}** particles and **{len(df_xy)}** outline points.")


# =============================================================================
# Module 1
# =============================================================================
if module == "Module 1":
    # Then at the bottom of EACH module page, just call:
    show_citation()
    st.header("Module 1 — Outline Reconstruction with EFA")

    with st.expander("📘 Instructions", expanded=True):
        st.write(
            "Choose **one** particle and harmonic orders from **1 to 40**. "
            "Observe how the outline converges to the original shape using inverse Fourier transform."
        )

    show_images(["Module1_1.png", "Module1_2.png"], "📌 EFA reconstruction method", expanded=EXPANDED, ncols=2, image_width=800)

    display_ids = [id_map[o] for o in original_ids]
    chosen_display = st.selectbox("Select ID", display_ids, index=0)
    chosen_original = reverse_map.get(chosen_display, chosen_display)

    g = df_xy[df_xy["ID"].astype(str) == str(chosen_original)]
    x = g["X"].to_numpy(dtype=float)
    y = g["Y"].to_numpy(dtype=float)

    if len(x) < MIN_POINTS_PER_PARTICLE:
        st.error(
            f"❌ Particle **{chosen_display}** has only **{len(x)}** point(s). "
            f"At least **{MIN_POINTS_PER_PARTICLE}** are required to build an outline. "
            "Please pick a different particle, or re-export this one with more points."
        )
        st.stop()

    area_px2 = float(OutlineArea(x, y))
    per_px = float(OutlineCircumference(x, y))

    c1, c2, c3 = st.columns([1.0, 1.0, 1.0])
    with c1:
        st.metric("Area (px²)", f"{area_px2:.4g}")
        area_unit2 = convert_px2_to_unit2(area_px2, px_per_mm, selected_unit)
        unit_label = get_unit_label(selected_unit, is_area=True)
        st.metric(f"Area ({unit_label})", f"{area_unit2:.4g}")
    with c2:
        st.metric("Perimeter (px)", f"{per_px:.4g}")
        per_unit = convert_px_to_unit(per_px, px_per_mm, selected_unit)
        unit_label_len = get_unit_label(selected_unit, is_area=False)
        st.metric(f"Perimeter ({unit_label_len})", f"{per_unit:.4g}")
    with c3:
        st.metric("px/mm", f"{px_per_mm:.4g}")
        st.metric("Unit", f"{unit_label_len}")

    maxK = 40
    K = st.slider("Harmonic order (K)", min_value=1, max_value=maxK, value=min(10, maxK))

    try:
        fig_rec, xt, yt = plot_reconstruction(x, y, int(K))
        r2, nrmse = contour_fit_metrics(x, y, xt, yt, m=200)

        m1, m2 = st.columns(2)
        m1.metric("Reconstruction Quality (R²)", f"{r2:.3f}")
        m2.metric("NRMSE", f"{nrmse:.4g}")
    except Exception as e:
        st.warning(f"Reconstruction metric failed: {e}")
        fig_rec = None

    with st.expander("Reconstruction of selected particle", expanded=EXPANDED):
        left, right = st.columns(2)
        with left:
            if fig_rec is not None:
                st.pyplot(fig_rec, width='stretch')
        with right:
            st.pyplot(plot_ellipse_and_circle(x, y), width='stretch')

    # Then at the bottom of EACH module page, just call:
    show_citation()

# =============================================================================
# Module 2
# =============================================================================
elif module == "Module 2":
    # Then at the bottom of EACH module page, just call:
    show_citation()
    st.header("Module 2 — Sensitivity Analysis")

    with st.expander("📘 Instructions", expanded=True):
        st.write(
            "Apply different harmonic orders to the whole dataset to analyze sensitivity of shape indices. "
            "Select multiple harmonic orders and observe how Asymmetricity and Polygonality values stabilize "
            "as harmonic order increases."
        )

    show_images(["Module2_1.png"], "📌 Asymmetricity and Polygonality equations",
                expanded=EXPANDED, ncols=1, image_width=400)

    max_h = 40
    default_N = [1, 2, 4, 8, 16, 32]
    chosen_N = st.multiselect(
        "Select harmonics order to evaluate",
        options=list(range(1, max_h + 1)),
        default=[n for n in default_N if n <= max_h],
    )
    chosen_N = sorted(set(int(n) for n in chosen_N))
    if not chosen_N:
        st.info("Choose at least one harmonic order to evaluate.")
        st.stop()

    current_key = (file_hash, tuple(chosen_N))

    if st.session_state.get("module2_key") == current_key:
        sens = st.session_state.get("module2_sens")
    else:
        sens = None
        st.info("Settings changed. Click **Run sensitivity** to compute.")

    @st.cache_data(show_spinner=False)
    def _compute_for_N(h: str, df_xy_local: pd.DataFrame, N: int) -> pd.DataFrame:
        p = ShapeParams(
            No_harmonics=int(N),
            no_sum=int(N),
            large_limit=PARAMS_FIXED.large_limit,
            small_limit=PARAMS_FIXED.small_limit,
            flag_location=PARAMS_FIXED.flag_location,
            flag_scale=PARAMS_FIXED.flag_scale,
            flag_rotation=PARAMS_FIXED.flag_rotation,
            flag_start=PARAMS_FIXED.flag_start,
        )
        res, _ = compute_shape_indices(df_xy_local, p)
        if res.empty:
            return pd.DataFrame()
        out = res[["ID", "asymmetricity2", "polygonality"]].copy()
        out = out.rename(columns={"asymmetricity2": "Asymmetricity", "polygonality": "Polygonality"})
        out["Harmonics"] = int(N)
        return out

    run = st.button("Run sensitivity", type="primary")
    if run:
        prog = st.progress(0)
        status = st.empty()

        rows = []
        total = len(chosen_N)
        for i, N in enumerate(chosen_N, start=1):
            status.write(f"Computing N = {N}  ({i}/{total})")
            part = _compute_for_N(file_hash, df_xy, int(N))
            if not part.empty:
                rows.append(part)
            prog.progress(int(i / max(total, 1) * 100))

        status.empty()
        prog.empty()

        if not rows:
            st.error("No sensitivity data produced.")
            st.stop()

        sens = pd.concat(rows, ignore_index=True)
        sens["Particle_ID"] = sens["ID"].map(id_map).fillna(sens["ID"])

        st.session_state["module2_sens"] = sens
        st.session_state["module2_key"] = current_key

    if sens is None:
        st.stop()

    sens_sel = sens[sens["Harmonics"].isin(chosen_N)].copy()

    def create_sensitivity_boxplot(data, y_col, title, color):
        fig, ax = plt.subplots(figsize=(6, 4))

        harmonics = sorted(data['Harmonics'].unique())
        box_data = [data[data['Harmonics'] == h][y_col].dropna() for h in harmonics]

        bp = ax.boxplot(box_data, positions=range(len(harmonics)),
                        patch_artist=True, widths=0.6)

        for box in bp['boxes']:
            box.set_facecolor(color)
            box.set_alpha(0.7)
            box.set_edgecolor('black')
            box.set_linewidth(1)

        for whisker in bp['whiskers']:
            whisker.set_color('black')
            whisker.set_linewidth(1)
        for cap in bp['caps']:
            cap.set_color('black')
            cap.set_linewidth(1)
        for median in bp['medians']:
            median.set_color('red')
            median.set_linewidth(2)
        for flier in bp['fliers']:
            flier.set_marker('o')
            flier.set_markerfacecolor(color)
            flier.set_markeredgecolor('black')
            flier.set_markersize(4)
            flier.set_alpha(0.5)

        ax.set_ylabel(y_col, fontsize=11)
        ax.set_title(f'{title} vs Harmonic Order', fontsize=12, fontweight='bold')
        ax.set_xlabel('Harmonic Order', fontsize=11)
        ax.set_xticks(range(len(harmonics)))
        ax.set_xticklabels([str(h) for h in harmonics])
        ax.grid(True, alpha=0.2, axis='y')

        plt.tight_layout()
        return fig

    with st.expander("Sensitivity Boxplots", expanded=EXPANDED):
        col1, col2 = st.columns(2)

        with col1:
            fig_asym = create_sensitivity_boxplot(
                sens_sel,
                'Asymmetricity',
                'Asymmetricity Sensitivity',
                '#FFD700'
            )
            st.pyplot(fig_asym, width='stretch')
            plt.close(fig_asym)

        with col2:
            fig_poly = create_sensitivity_boxplot(
                sens_sel,
                'Polygonality',
                'Polygonality Sensitivity',
                '#2E8B57'
            )
            st.pyplot(fig_poly, width='stretch')
            plt.close(fig_poly)

    with st.expander("📉 Stabilization Trend (Median Values)", expanded=False):
        trend = sens.groupby("Harmonics")[["Asymmetricity", "Polygonality"]].median().reset_index()

        fig_trend, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(trend['Harmonics'], trend['Asymmetricity'], 'o-', color='#FFD700',
                 linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
        ax1.set_xlabel('Harmonic Order', fontsize=11)
        ax1.set_ylabel('Median Asymmetricity', fontsize=11)
        ax1.set_title('Asymmetricity Stabilization', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.2)
        ax1.set_xticks(trend['Harmonics'])

        ax2.plot(trend['Harmonics'], trend['Polygonality'], 'o-', color='#2E8B57',
                 linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
        ax2.set_xlabel('Harmonic Order', fontsize=11)
        ax2.set_ylabel('Median Polygonality', fontsize=11)
        ax2.set_title('Polygonality Stabilization', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.2)
        ax2.set_xticks(trend['Harmonics'])

        plt.tight_layout()
        st.pyplot(fig_trend, width='stretch')
        plt.close(fig_trend)

        st.markdown("**Stabilization Analysis:**")
        col1, col2 = st.columns(2)

        with col1:
            asym_diff = trend['Asymmetricity'].diff().abs().dropna()
            if len(asym_diff) > 0:
                stable_k = trend.iloc[asym_diff.idxmin()]['Harmonics']
                st.info(f"📊 Asymmetricity stabilizes around K = {int(stable_k)}")

        with col2:
            poly_diff = trend['Polygonality'].diff().abs().dropna()
            if len(poly_diff) > 0:
                stable_k = trend.iloc[poly_diff.idxmin()]['Harmonics']
                st.info(f"📐 Polygonality stabilizes around K = {int(stable_k)}")

    with st.expander("🧾 Sensitivity Table", expanded=EXPANDED):
        st.dataframe(
            sens_sel[["Particle_ID", "Harmonics", "Asymmetricity", "Polygonality"]]
            .sort_values(["Harmonics", "Particle_ID"]),
            use_container_width=True,
        )

    st.download_button(
        "⬇️ Download sensitivity table",
        data=sens_sel[["Particle_ID", "Harmonics", "Asymmetricity", "Polygonality"]].to_csv(index=False).encode("utf-8"),
        file_name="sensitivity_table.csv",
        mime="text/csv",
    )
    # Then at the bottom of EACH module page, just call:
    show_citation()

# =============================================================================
# Module 3
# =============================================================================
else:
    st.header("Module 3 — Comprehensive Shape Statistics")

    with st.expander("📘 Instructions", expanded=True):
        st.write(
            "Compute all shape indices using EFA, plot distributions, and analyze correlations "
            "to find redundant descriptors. Select the optimal harmonic order based on your "
            "sensitivity analysis results from Module 2."
        )

    show_images(
        ["Module3_1.png", "Module3_2.png", "Module3_3.png", "Module3_4.png"],
        "📌 Module 3 statistical analysis methods",
        expanded=EXPANDED,
        ncols=2,
        image_width=250
    )

    st.subheader("⚙️ Analysis Settings")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Select optimal harmonics order from sensitivity analysis results**")
    with col2:
        selected_harmonics = st.number_input(
            "Harmonics order",
            min_value=1,
            max_value=64,
            value=PARAMS_FIXED.No_harmonics,
            step=1
        )

    current_params = ShapeParams(
        No_harmonics=selected_harmonics,
        no_sum=selected_harmonics,
        large_limit=PARAMS_FIXED.large_limit,
        small_limit=PARAMS_FIXED.small_limit,
        flag_location=PARAMS_FIXED.flag_location,
        flag_scale=PARAMS_FIXED.flag_scale,
        flag_rotation=PARAMS_FIXED.flag_rotation,
        flag_start=PARAMS_FIXED.flag_start,
    )

    key3 = (file_hash, float(px_per_mm), selected_harmonics)

    if st.session_state.get("module3_key") == key3:
        results_df = st.session_state.get("module3_results")
        errors_df = st.session_state.get("module3_errors")
    else:
        results_df = None
        errors_df = None
        st.info("Settings changed. Click **Run EFA on whole dataset** to compute.")

    run3 = st.button("Run EFA on whole dataset", type="primary")
    if run3:
        with st.spinner(f"Computing shape indices for all particles with {selected_harmonics} harmonics..."):
            res, err = compute_shape_indices(df_xy, current_params)

        if res.empty:
            st.error("No results produced.")
            st.stop()

        res = res.copy()
        res["Particle_ID"] = res["ID"].map(id_map).fillna(res["ID"])

        res["area_converted"] = res["area"].apply(lambda v: convert_px2_to_unit2(v, px_per_mm, selected_unit))
        res["perimeter_converted"] = res["perimeter"].apply(lambda v: convert_px_to_unit(v, px_per_mm, selected_unit))

        st.session_state["selected_unit"] = selected_unit
        st.session_state["unit_label_area"] = get_unit_label(selected_unit, is_area=True)
        st.session_state["unit_label_len"] = get_unit_label(selected_unit, is_area=False)
        st.session_state["module3_results"] = res
        st.session_state["module3_errors"] = err
        st.session_state["module3_key"] = key3

        results_df = res
        errors_df = err

    if results_df is None:
        st.stop()

    st.subheader("Results - all shape indices")
    st.dataframe(results_df, use_container_width=True)

    st.download_button(
        "⬇️ Download full results (CSV)",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="shape_indices_outlines.csv",
        mime="text/csv",
    )

    if errors_df is not None and not errors_df.empty:
        with st.expander("Errors", expanded=False):
            st.dataframe(errors_df, use_container_width=True)

    with st.expander("📊 Distribution Analysis (Histogram + KDE + Boxplot)", expanded=EXPANDED):
        unit_label_area = st.session_state.get("unit_label_area", "mm²")
        unit_label_len = st.session_state.get("unit_label_len", "mm")

        display_names = {
            "area_converted": f"Area ({unit_label_area})",
            "perimeter_converted": f"Perimeter ({unit_label_len})",
            "Kael": "Elongation Ratio",
            "angularity": "Angularity",
            "surface_roughness": "Surface Roughness",
            "asymmetricity1": "Asymmetricity",
            "asymmetricity2": "Normalized Asymmetricity",
            "polygonality": "Polygonality",
            "elongation": "Geometric Elongation",
            "bulkiness": "Bulkiness",
            "surface": "Surface Factor",
            "circularity": "Circularity",
            "sphericity": "Sphericity"
        }

        dist_cols = [
            ("area_converted", f"Area ({unit_label_area})"),
            ("perimeter_converted", f"Perimeter ({unit_label_len})"),
            ("Kael", "Elongation Ratio"),
            ("angularity", "Angularity"),
            ("surface_roughness", "Surface Roughness"),
            ("asymmetricity1", "Asymmetricity"),
            ("asymmetricity2", "Normalized Asymmetricity"),
            ("polygonality", "Polygonality"),
            ("elongation", "Geometric Elongation"),
            ("bulkiness", "Bulkiness"),
            ("surface", "Surface Factor"),
            ("circularity", "Circularity"),
            ("sphericity", "Sphericity")
        ]

        def enhanced_distribution_plot(data, col, display_name):
            fig = plt.figure(figsize=(8, 6), dpi=150)
            gs = fig.add_gridspec(4, 1, hspace=0.3, height_ratios=[3, 1, 3, 1])

            ax1 = fig.add_subplot(gs[0, 0])
            values = data[col].dropna()

            if len(values) == 0:
                ax1.text(0.5, 0.5, "No data available", ha="center", va="center")
                ax1.set_title(f'{display_name} Distribution', fontsize=12, fontweight='bold')
                return fig

            n, bins, patches = ax1.hist(values, bins=20, density=True, alpha=0.7,
                                        color='steelblue', edgecolor='black', linewidth=0.5)

            from scipy import stats
            kde = stats.gaussian_kde(values)
            x_range = np.linspace(values.min(), values.max(), 200)
            ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

            mean_val = values.mean()
            median_val = values.median()
            ax1.axvline(mean_val, color='green', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.3f}')
            ax1.axvline(median_val, color='orange', linestyle=':', linewidth=1.5, label=f'Median: {median_val:.3f}')

            ax1.set_ylabel('Density')
            ax1.set_title(f'{display_name} Distribution', fontsize=12, fontweight='bold')
            ax1.legend(loc='upper right', fontsize=8)
            ax1.grid(True, alpha=0.2)

            ax2 = fig.add_subplot(gs[1, 0])
            bp = ax2.boxplot(values, vert=False, patch_artist=True, widths=0.6)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][0].set_alpha(0.7)
            bp['whiskers'][0].set_color('black')
            bp['whiskers'][1].set_color('black')
            bp['medians'][0].set_color('red')
            bp['medians'][0].set_linewidth(2)
            ax2.set_xlabel(display_name)
            ax2.set_yticks([])
            ax2.grid(True, alpha=0.2, axis='x')

            return fig

        cols = st.columns(2)
        plot_count = 0

        for col, display_name in dist_cols:
            if col in results_df.columns and not results_df[col].isna().all():
                with cols[plot_count % 2]:
                    fig = enhanced_distribution_plot(results_df, col, display_name)
                    st.pyplot(fig, width='stretch')
                    plt.close(fig)

                    stats_data = results_df[col].describe(percentiles=[.25, .5, .75])
                    stats_df = pd.DataFrame({
                        'Statistic': ['Count', 'Mean', 'Std', 'Min', '25%', 'Median', '75%', 'Max'],
                        'Value': [
                            f"{stats_data['count']:.0f}",
                            f"{stats_data['mean']:.4f}",
                            f"{stats_data['std']:.4f}",
                            f"{stats_data['min']:.4f}",
                            f"{stats_data['25%']:.4f}",
                            f"{stats_data['50%']:.4f}",
                            f"{stats_data['75%']:.4f}",
                            f"{stats_data['max']:.4f}"
                        ]
                    })
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                plot_count += 1

    with st.expander("📋 Summary Statistics", expanded=EXPANDED):
        stat_cols = ["area_converted", "perimeter_converted"] + [col for col, _ in dist_cols if col not in ["area_converted", "perimeter_converted"]]
        stat_cols = [c for c in stat_cols if c in results_df.columns]

        stats_data = []
        for col in stat_cols:
            display_name = display_names.get(col, col)
            desc = results_df[col].describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90])
            stats_data.append({
                'Descriptor': display_name,
                'Count': f"{desc['count']:.0f}",
                'Mean': f"{desc['mean']:.4f}",
                'Std': f"{desc['std']:.4f}",
                'Min': f"{desc['min']:.4f}",
                '10%': f"{desc['10%']:.4f}",
                '25%': f"{desc['25%']:.4f}",
                'Median': f"{desc['50%']:.4f}",
                '75%': f"{desc['75%']:.4f}",
                '90%': f"{desc['90%']:.4f}",
                'Max': f"{desc['max']:.4f}"
            })

        stats_tbl = pd.DataFrame(stats_data)
        st.dataframe(stats_tbl, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download summary statistics (CSV)",
            data=stats_tbl.to_csv(index=False).encode("utf-8"),
            file_name="summary_statistics.csv",
            mime="text/csv",
        )

    with st.expander("🔗 Correlation Matrix", expanded=EXPANDED):
        corr_cols = stat_cols

        corr_df = results_df[corr_cols].copy()
        corr_df.rename(columns=display_names, inplace=True)

        corr = corr_df.apply(pd.to_numeric, errors="coerce").corr(method="pearson")

        st.download_button(
            "⬇️ Download correlation matrix (CSV)",
            data=corr.to_csv().encode("utf-8"),
            file_name="correlation_matrix.csv",
            mime="text/csv",
        )

        fig_corr = px.imshow(
            corr,
            aspect="auto",
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            text_auto=".2f",
            title="Correlation Matrix (Pearson)",
        )

        mask = np.triu(np.ones_like(corr), k=1).astype(bool)

        fig_corr.update_traces(
            z=np.where(mask, np.nan, corr.values),
            text=np.where(mask, "", np.round(corr.values, 2).astype(str))
        )

        fig_corr.update_xaxes(side="bottom", tickangle=45)
        fig_corr.update_layout(
            height=600,
            width=600,
            xaxis_title="",
            yaxis_title=""
        )
        st.plotly_chart(fig_corr, width='stretch')

        st.markdown("### 🔍 Key Observations")

        observations = []
        corr_values = []

        for i, row in enumerate(corr.index):
            for j, col in enumerate(corr.columns):
                if i > j:
                    val = corr.iloc[i, j]
                    if not np.isnan(val):
                        strength = "Very strong" if abs(val) >= 0.8 else \
                            "Strong" if abs(val) >= 0.6 else \
                                "Moderate" if abs(val) >= 0.4 else \
                                    "Weak" if abs(val) >= 0.2 else "Very weak"
                        direction = "positive" if val > 0 else "negative"
                        observations.append(f"- {strength} {direction} correlation between **{row}** and **{col}** ({val:.2f})")
                        corr_values.append(abs(val))

        sorted_obs = [x for _, x in sorted(zip(corr_values, observations), reverse=True)]

        st.markdown("**Top strongest correlations:**")
        for obs in sorted_obs[:8]:
            st.markdown(obs)

        st.markdown("\n**Indices with generally weak correlations to others:**")
        weak_indices = []
        for idx in corr.index:
            idx_corrs = [corr.loc[idx, col] for col in corr.columns if col != idx and not np.isnan(corr.loc[idx, col])]
            if idx_corrs:
                avg_abs_corr = np.mean([abs(c) for c in idx_corrs])
                if avg_abs_corr < 0.3:
                    weak_indices.append(f"- **{idx}** (average |r| = {avg_abs_corr:.2f})")

        if weak_indices:
            for weak_idx in weak_indices[:5]:
                st.markdown(weak_idx)
        else:
            st.markdown("- No indices with consistently weak correlations found")

        st.markdown("""
        ---
        **Correlation Interpretation (based on absolute value):**
        - **|r| ≥ 0.8**: Very strong correlation (positive or negative)
        - **0.6 ≤ |r| < 0.8**: Strong correlation
        - **0.4 ≤ |r| < 0.6**: Moderate correlation
        - **0.2 ≤ |r| < 0.4**: Weak correlation
        - **|r| < 0.2**: Very weak or no correlation

        *Note: Negative values indicate inverse relationships (as one increases, the other decreases)*
        """)
