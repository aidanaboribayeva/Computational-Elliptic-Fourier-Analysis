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
                # Use width parameter to control image size
                st.image(str(p), width=image_width)


# =============================================================================
# CSV loader
# =============================================================================
def load_xy_csv(file_bytes: bytes) -> pd.DataFrame:
    """Robust loader for ID, X, Y (handles , ; tab)."""
    df = None
    for sep in [",", ";", "\t", None]:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, engine="python")
            if df is not None and df.shape[1] >= 3:
                break
        except Exception:
            df = None

    if df is None or df.empty:
        raise ValueError("Could not read CSV. Expected columns: ID, X, Y.")

    df = df.rename(columns={c: str(c).strip() for c in df.columns})
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
        cols = list(df.columns[:3])
        id_col, x_col, y_col = cols[0], cols[1], cols[2]

    out = df[[id_col, x_col, y_col]].copy()
    out.columns = ["ID", "X", "Y"]
    out["ID"] = out["ID"].astype(str).str.strip()
    out["X"] = pd.to_numeric(out["X"], errors="coerce")
    out["Y"] = pd.to_numeric(out["Y"], errors="coerce")
    out = out.dropna(subset=["ID", "X", "Y"]).reset_index(drop=True)
    return out


def make_id_mapping(original_ids: List[str]) -> Dict[str, str]:
    return {oid: f"P{i:03d}" for i, oid in enumerate(original_ids, start=1)}


# =============================================================================
# Units (px/mm)
# =============================================================================
def px_to_mm(length_px: float, px_per_mm: float) -> float:
    px_per_mm = float(max(px_per_mm, 1e-12))
    return float(length_px) / px_per_mm


def px2_to_mm2(area_px2: float, px_per_mm: float) -> float:
    px_per_mm = float(max(px_per_mm, 1e-12))
    return float(area_px2) / (px_per_mm ** 2)


# =============================================================================
# Geometry helpers (your implementations)
# =============================================================================
def _ensure_closed(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) >= 3 and (x[0] != x[-1] or y[0] != y[-1]):
        x = np.append(x, x[0])
        y = np.append(y, y[0])
    return x, y


def OutlineArea(x, y):
    """
    Calculates the area of a closed polygon outline using the vectorized Shoelace Formula.
    """
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
    """
    Calculates the perimeter (circumference) of a closed polygon outline
    using vectorized NumPy operations.
    """
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
    s = 1.  # s=dsin(pi2)
    cc = 0.  # cc=dcos(pi2)*dcos(pi2)
    Q = (1. - s * ak) * (1. + s * ak)
    ELLE_val = s * (elliprf(cc, Q, 1.) - ((s * ak) * (s * ak)) * elliprd(cc, Q, 1.) / 3.)
    return ELLE_val


def ComputeEllFourierCoef(No_OutlinePoints, X, Y, No_harmonics):
    """
    Computes Elliptic Fourier Descriptors (EFDs) for a closed contour.

    Arguments:
        No_OutlinePoints (int): Number of points in the contour (N).
        X (array-like): X-coordinates of the outline points (length N).
        Y (array-like): Y-coordinates of the outline points (length N).
        No_harmonics (int): Number of harmonics to calculate (N_H).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: (Ax, Bx, Ay, By)
        Ax[0] and Ay[0] hold the DC components (Ax0, Ay0).
        Indices 1 to N_H hold the harmonic coefficients.
    """
    X = asarray(X, dtype=float64)
    Y = asarray(Y, dtype=float64)
    X_shifted = roll(X, -1)
    Y_shifted = roll(Y, -1)
    DXi = X_shifted - X
    DYi = Y_shifted - Y
    DTi = sqrt(DXi ** 2 + DYi ** 2)
    DTi = where(DTi == 0, 1e-12, DTi)  # Avoid division by zero
    R_x = DXi / DTi
    R_y = DYi / DTi
    T_curr = np.cumsum(DTi)
    T_prev = np.concatenate(([0.0], T_curr[:-1]))
    PolygonLength = T_curr[-1]  # T: The total perimeter length

    # Calculate DC components (Ax0, Ay0)
    Tsum = 0.0
    Xsum = 0.0  # Sum of Delta X up to segment i-1
    Ysum = 0.0  # Sum of Delta Y up to segment i-1
    Ax0_integral_sum = 0.0
    Ay0_integral_sum = 0.0

    for i in range(No_OutlinePoints):
        DT_i = DTi[i]
        Tnew = Tsum + DT_i
        #
        Rx_i = R_x[i]
        Ry_i = R_y[i]
        #
        Xi_i = Xsum - Rx_i * Tsum
        Delta_i = Ysum - Ry_i * Tsum
        #
        Ax0_integral_sum += 0.5 * Rx_i * (Tnew * Tnew - Tsum * Tsum) + Xi_i * DT_i
        Ay0_integral_sum += 0.5 * Ry_i * (Tnew * Tnew - Tsum * Tsum) + Delta_i * DT_i
        #
        Tsum = Tnew
        Xsum += DXi[i]
        Ysum += DYi[i]
        #
    if PolygonLength > 0.0:
        Ax0 = X[0] + Ax0_integral_sum / PolygonLength
        Ay0 = Y[0] + Ay0_integral_sum / PolygonLength
    else:
        Ax0 = X[0]
        Ay0 = Y[0]

    # Initialize coefficient arrays
    Ax = zeros(No_harmonics + 1)
    Bx = zeros(No_harmonics + 1)
    Ay = zeros(No_harmonics + 1)
    By = zeros(No_harmonics + 1)

    Ax[0] = Ax0
    Ay[0] = Ay0

    if No_harmonics == 0 or PolygonLength == 0.0:
        return (Ax, Bx, Ay, By)

    # Calculate harmonic coefficients
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
    """
    Estimates X,Y-coordinates from Elliptic Fourier coefficients using NumPy vectorization.

    Arguments:
        Ax (np.ndarray): Fourier coefficients for X-coordinate (Ax[1:] are for harmonics 1 to N_harmonics).
        Bx (np.ndarray): Fourier coefficients for X-coordinate (harmonics 1 to N_harmonics).
        Ay (np.ndarray): Fourier coefficients for Y-coordinate (Ay[1:] are for harmonics 1 to N_harmonics).
        By (np.ndarray): Fourier coefficients for Y-coordinate (harmonics 1 to N_harmonics).
        No_XY (int): Number of X,Y coordinates to be estimated.

    Returns:
        Estimated (Xest, Yest) coordinates.
    """
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
    """
    Normalize Fourier coefficients for location, scale, rotation, and starting point.
    """
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
    """
    Wrapper function for coefficient normalization that matches your original code structure.
    """
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
    """
    ae, be  - major and minor ellipse semi-axis
    K       - shape index (elongation ratio)
    Le      - circumference of approximated ellipse
    rc      - radius of circle whose perimeter is equivalent to that of ellipse
    Sr      - standard deviation of radial distance from centroid to actual
              particle outline, rd, and that of approximated ellipse, re
    Uc      - unevenness coefficient, Uc=Sr/rc
    S1, S2  - sharpness coefficients
    """
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

    # Fix for division by zero warning
    with np.errstate(divide='ignore', invalid='ignore'):
        di = where(np.abs(dx) <= 1E-30, 1000.0, dy / dx)
        # Replace any inf or nan values with 1000.0
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
    """
    Calculate shape indices from elliptic Fourier coefficients.
    """
    FlagRotation = 1  # Default value
    psi = RotateAngle

    # First, get the basic shape indices from ShapeIndex
    ae, be, k, Le, rc, Uc, S1, S2, Bk1, Bk2, Ask, Bsk = ShapeIndex(
        Ax, Bx, Ay, By, FlagRotation, psi, XMID, YMID, Scale1, X, Y, No_sum, LargeLimit
    )

    # IMPORTANT: In the original code, coefficients are recalculated here
    Ax, Bx, Ay, By = ComputeEllFourierCoef(No_OutlinePoints, X, Y, No_harmonics)

    # Calculate Asymmetricity2 (Ass2)
    NoAss = min(No_harmonics, No_sum)
    Ax_slice = Ax[1:NoAss + 1]
    Bx_slice = Bx[1:NoAss + 1]
    Ay_slice = Ay[1:NoAss + 1]
    By_slice = By[1:NoAss + 1]

    abs_Ax = abs(Ax_slice)
    abs_Bx = abs(Bx_slice)
    abs_Ay = abs(Ay_slice)
    abs_By = abs(By_slice)

    # Calculate SumBx (sum of |Bx|/|Ax| where both > SmallLimit)
    condition_Bx = (abs_Bx > SmallLimit) & (abs_Ax > SmallLimit)
    ratio_Bx_Ax = np.divide(abs_Bx, abs_Ax, out=np.zeros_like(abs_Bx), where=condition_Bx)
    SumBx = np.sum(ratio_Bx_Ax)

    # Calculate SumAy (sum of |Ay|/|By| where both > SmallLimit)
    condition_Ay = (abs_Ay > SmallLimit) & (abs_By > SmallLimit)
    ratio_Ay_By = np.divide(abs_Ay, abs_By, out=np.zeros_like(abs_Ay), where=condition_Ay)
    SumAy = np.sum(ratio_Ay_By)

    Ass2 = np.sqrt(SumBx * SumAy) if (SumBx * SumAy) > 0 else 0.0

    # Calculate Asymmetricity1 (Ass1) - using harmonics 2 to 15
    # Only if we have at least 2 harmonics
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

        # Calculate Polygonality (P) - only if we have at least 2 harmonics
        # Initialize with default values
        NoMaxAx1 = 1
        NoMaxBy1 = 1
        MaxAx = abs(Ax[1])
        MaxBy = abs(By[1])

        # Find first maximum
        j = 1
        while j <= No_harmonics:
            if MaxAx < abs(Ax[j]):
                MaxAx = abs(Ax[j])
                NoMaxAx1 = j
            if MaxBy < abs(By[j]):
                MaxBy = abs(By[j])
                NoMaxBy1 = j
            j = j + 1

        # Find second maximum
        j = 1
        MaxAx2 = -1  # Start with a very small value
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
        # For single harmonic, set default values
        Ass1 = 0.0
        P = 1.0  # Minimum polygonality

    return (ae, be, k, Le, rc, Uc, S1, S2, Bk1, Bk2, Ask, Bsk, Ass1, Ass2, P)


def ComputeAngularity(Ax, Bx, Ay, By, No_harmonics, w=360):
    """
    Compute angularity index (AIg) using Fourier coefficients
    """
    if No_harmonics < 1 or len(Ax) < 2:
        return 0.0  # Return default value if not enough harmonics

    TwoPi = 2.0 * pi
    angles = np.linspace(0, TwoPi, w, endpoint=False)
    n_array = np.arange(1, No_harmonics + 1)

    # Create the matrix for n*u (angles x harmonics)
    n_u = np.outer(angles, n_array)
    sin_n_u = np.sin(n_u)
    cos_n_u = np.cos(n_u)

    # Get coefficients for harmonics 1 to No_harmonics
    Ax_n = Ax[1:No_harmonics + 1]
    Bx_n = Bx[1:No_harmonics + 1]
    Ay_n = Ay[1:No_harmonics + 1]
    By_n = By[1:No_harmonics + 1]

    # Multiply by harmonic number
    nAx = n_array * Ax_n
    nBx = n_array * Bx_n
    nAy = n_array * Ay_n
    nBy = n_array * By_n

    # Calculate derivatives
    x_deriv = np.sum(-nAx[:, np.newaxis] * sin_n_u.T + nBx[:, np.newaxis] * cos_n_u.T, axis=0)
    y_deriv = np.sum(-nAy[:, np.newaxis] * sin_n_u.T + nBy[:, np.newaxis] * cos_n_u.T, axis=0)

    hx = np.arctan2(y_deriv, x_deriv)
    hx = np.unwrap(hx)
    dh = np.abs(np.diff(hx, append=hx[0]))
    AIg = (1.0 / TwoPi) * np.sum(dh) - 1.0
    return AIg


def ShapeFactors(NoP, X, Y, Area, Perimeter):
    """
    Calculate shape factors: Elongation, Bulkiness, Surface, Circularity, Sphericity
    """
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
# Fixed computation params (matching your original code)
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
# Batch computation (matching your original code structure)
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

            # Ensure closed contour
            X, Y = _ensure_closed(X, Y)
            NoP = len(X)

            # Calculate centroid
            XMID, YMID = OutlineCentroid(X, Y)

            # Compute normalized Fourier coefficients (these will be used for ShapeIndex only)
            Ax, Bx, Ay, By, Scale1, RotateAngle, StartAngle = CoefNormalization(
                X, Y, XMID, YMID,
                params.flag_location,
                params.flag_scale,
                params.flag_rotation,
                params.flag_start,
                params.No_harmonics
            )

            # Calculate shape indices from EFA
            # Note: ShapeIndicesEF will recalculate coefficients internally
            ae, be, k, Le, rc, Uc, S1, S2, Bk1, Bk2, Ask, Bsk, Ass1, Ass2, P = ShapeIndicesEF(
                Ax, Bx, Ay, By,  # These are passed but will be recalculated inside
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

            # Calculate angularity using the original coefficients
            # We need to recalculate coefficients for angularity too
            Ax_raw, Bx_raw, Ay_raw, By_raw = ComputeEllFourierCoef(NoP, X, Y, params.No_harmonics)
            AIg = ComputeAngularity(Ax_raw, Bx_raw, Ay_raw, By_raw, No_harmonics=params.no_sum, w=360)

            # Basic geometry
            area = OutlineArea(X, Y)
            perimeter = OutlineCircumference(X, Y)

            # Calculate shape factors
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
    """Plot original outline and reconstruction using inverse Fourier transform."""
    x, y = _ensure_closed(x, y)
    NoP = len(x)

    # Compute coefficients
    Ax, Bx, Ay, By = ComputeEllFourierCoef(NoP, x, y, K)

    # Reconstruct with same number of points
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
    """
    Covariance-based ellipse orientation + scale to match polygon area.
    Returns (cx, cy, a, b, theta) with a>=b.
    """
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
    """Plot equal-area ellipse and circle."""
    x, y = _ensure_closed(x, y)

    # Get ellipse parameters and circle radius using the existing functions
    cx, cy, a, b, theta = fit_ellipse_equal_area(x, y)
    r = equal_area_circle_radius(x, y)

    # Generate ellipse points
    t = np.linspace(0, 2 * np.pi, 300)
    ex = cx + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
    ey = cy + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)

    # Generate circle points
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

    # Optional debug info
    print(f"Ellipse: a={a:.2f}, b={b:.2f}, ratio={b / a:.3f}")
    print(f"Circle radius: r={r:.2f}")
    print(f"Ellipse area (π*a*b): {np.pi * a * b:.2f}")
    print(f"Original area: {OutlineArea(x, y):.2f}")

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
    """Return (R², NRMSE)."""
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
# Figure_5-style histogram helper (percentiles legend)
# =============================================================================
def percentile_histogram(df: pd.DataFrame, col: str, xlabel: str) -> plt.Figure:
    data = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.0, 4.0), dpi=170)

    if data.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    p10 = float(np.percentile(data, 10))
    p50 = float(np.percentile(data, 50))
    p90 = float(np.percentile(data, 90))

    ax.hist(data, bins=30, density=True, alpha=0.7, edgecolor="black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability")
    ax.legend([f"10% = {p10:.3f}\n50% = {p50:.3f}\n90% = {p90:.3f}"], fontsize=11, frameon=True)
    ax.grid(True, alpha=0.20)
    fig.tight_layout()
    return fig


def general_statistics_table(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    desc = numeric.describe(percentiles=[0.10, 0.50, 0.90]).T
    desc = desc.rename(
        columns={
            "count": "count",
            "mean": "mean",
            "std": "std",
            "min": "min",
            "10%": "p10",
            "50%": "p50",
            "90%": "p90",
            "max": "max",
        }
    )
    return desc.reset_index().rename(columns={"index": "descriptor"})


# =============================================================================
# Streamlit UI
# =============================================================================
st.set_page_config(page_title="Shape Analysis", layout="wide")
st.title("🔬 Shape Analysis with Elliptic Fourier Descriptors")

uploaded = st.sidebar.file_uploader("Upload CSV (ID, X, Y)", type=["csv"])

# # Font size control
# font_size = st.sidebar.slider("Text size", min_value=14, max_value=24, value=18, step=1)
# st.markdown(
#     f"""
#     <style>
#       html, body, [class*="css"] {{ font-size: {font_size}px !important; }}
#       section[data-testid="stSidebar"] * {{ font-size: {font_size}px !important; }}
#       .stMetricValue {{ font-size: {min(font_size + 12, 34)}px !important; }}
#       .stMetricLabel {{ font-size: {max(font_size - 2, 12)}px !important; }}
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

EXPANDED = True

st.sidebar.markdown("---")
module = st.sidebar.radio("Modules", ["Module 1", "Module 2", "Module 3"], index=0)

st.sidebar.markdown("---")
with st.sidebar.expander("📏 Set Scale", expanded=True):
    px_per_mm = st.number_input("px/mm", min_value=0.0001, value=77.0, step=0.1)

if uploaded is None:
    st.info("⬅️ Upload a CSV to start.")
    st.caption("Expected columns: ID, X, Y.")
    st.stop()

file_bytes = uploaded.getvalue()
file_hash = hashlib.md5(file_bytes).hexdigest()


@st.cache_data(show_spinner=False)
def _load_cached(h: str, b: bytes) -> pd.DataFrame:
    return load_xy_csv(b)


df_xy = _load_cached(file_hash, file_bytes)
if df_xy.empty:
    st.error("CSV loaded but has no valid rows for ID/X/Y.")
    st.stop()

original_ids = df_xy["ID"].dropna().astype(str).unique().tolist()
id_map = make_id_mapping(original_ids)
reverse_map = {v: k for k, v in id_map.items()}

st.caption(f"Detected **{df_xy['ID'].nunique()}** particles and **{len(df_xy)}** outline points.")

# =============================================================================
# Module 1
# =============================================================================
if module == "Module 1":
    st.header("Module 1 — Outline Reconstruction with EFA")

    with st.expander("📘 Instructions", expanded=True):
        st.write(
            "Choose **one** particle and harmonic orders from **1 to 40**. "
            "Observe how the outline converges to the original shape using inverse Fourier transform."
        )

    # Module 1 instruction figures
    show_images(["Module1_1.png", "Module1_2.png"], "📌 EFA reconstruction method", expanded=EXPANDED, ncols=2, image_width=800)

    display_ids = [id_map[o] for o in original_ids]
    chosen_display = st.selectbox("Select ID", display_ids, index=0)
    chosen_original = reverse_map.get(chosen_display, chosen_display)

    g = df_xy[df_xy["ID"].astype(str) == str(chosen_original)]
    x = g["X"].to_numpy(dtype=float)
    y = g["Y"].to_numpy(dtype=float)

    if len(x) < 3:
        st.error("This particle has <3 points.")
        st.stop()

    area_px2 = float(OutlineArea(x, y))
    per_px = float(OutlineCircumference(x, y))

    c1, c2, c3 = st.columns([1.0, 1.0, 1.0])
    with c1:
        st.metric("Area (px²)", f"{area_px2:.4g}")
        st.metric("Area (mm²)", f"{px2_to_mm2(area_px2, px_per_mm):.4g}")
    with c2:
        st.metric("Perimeter (px)", f"{per_px:.4g}")
        st.metric("Perimeter (mm)", f"{px_to_mm(per_px, px_per_mm):.4g}")
    with c3:
        st.metric("px/mm", f"{px_per_mm:.4g}")

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


# =============================================================================
# Module 2
# =============================================================================
elif module == "Module 2":
    st.header("Module 2 — Sensitivity Analysis")

    with st.expander("📘 Instructions", expanded=True):
        st.write(
            "Apply different harmonic orders to the whole dataset to analyze sensitivity of shape indices. "
            "Select multiple harmonic orders and observe how Asymmetricity and Polygonality values stabilize "
            "as harmonic order increases."
        )

    # Module 2 instruction figure
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


    # Function to create simplified boxplot with matplotlib (no mean/median markers)
    def create_sensitivity_boxplot(data, y_col, title, color):
        fig, ax = plt.subplots(figsize=(6, 4))

        # Prepare data for boxplot
        harmonics = sorted(data['Harmonics'].unique())
        box_data = [data[data['Harmonics'] == h][y_col].dropna() for h in harmonics]

        # Create boxplot
        bp = ax.boxplot(box_data, positions=range(len(harmonics)),
                        patch_artist=True, widths=0.6)

        # Color the boxes
        for box in bp['boxes']:
            box.set_facecolor(color)
            box.set_alpha(0.7)
            box.set_edgecolor('black')
            box.set_linewidth(1)

        # Style whiskers, caps, medians
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
        # Create two columns for the plots
        col1, col2 = st.columns(2)

        with col1:
            # Asymmetricity boxplot with yellow/gold
            fig_asym = create_sensitivity_boxplot(
                sens_sel,
                'Asymmetricity',
                'Asymmetricity Sensitivity',
                '#FFD700'  # Yellow/gold
            )
            st.pyplot(fig_asym, width='stretch')
            plt.close(fig_asym)

        with col2:
            # Polygonality boxplot with sea green
            fig_poly = create_sensitivity_boxplot(
                sens_sel,
                'Polygonality',
                'Polygonality Sensitivity',
                '#2E8B57'  # Sea green
            )
            st.pyplot(fig_poly, width='stretch')
            plt.close(fig_poly)

    with st.expander("📉 Stabilization Trend (Median Values)", expanded=False):
        trend = sens.groupby("Harmonics")[["Asymmetricity", "Polygonality"]].median().reset_index()

        # Create matplotlib line plots
        fig_trend, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Asymmetricity trend
        ax1.plot(trend['Harmonics'], trend['Asymmetricity'], 'o-', color='#FFD700',
                 linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
        ax1.set_xlabel('Harmonic Order', fontsize=11)
        ax1.set_ylabel('Median Asymmetricity', fontsize=11)
        ax1.set_title('Asymmetricity Stabilization', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.2)
        ax1.set_xticks(trend['Harmonics'])

        # Polygonality trend
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

        # Identify stabilization point
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
        data=sens_sel[["Particle_ID", "Harmonics", "Asymmetricity", "Polygonality"]].to_csv(index=False).encode(
            "utf-8"),
        file_name="sensitivity_table.csv",
        mime="text/csv",
    )

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

    # Module 3 instruction figures
    show_images(
        ["Module3_1.png", "Module3_2.png", "Module3_3.png", "Module3_4.png"],
        "📌 Multiscale shape characterization",
        expanded=EXPANDED,
        ncols=2, image_width=800
    )

    # Add harmonic order selection before running EFA
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

    # Update params with selected harmonics
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

        # Add unit conversions
        res["area_mm2"] = res["area"].apply(lambda v: px2_to_mm2(v, px_per_mm))
        res["perimeter_mm"] = res["perimeter"].apply(lambda v: px_to_mm(v, px_per_mm))

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

    # Enhanced distribution plots with KDE and boxplots
    with st.expander("📊 Distribution Analysis (Histogram + KDE + Boxplot)", expanded=EXPANDED):
        # Define display names mapping
        display_names = {
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


        # Function to create enhanced distribution plot with KDE and boxplot subplot
        def enhanced_distribution_plot(data, col, display_name):
            fig = plt.figure(figsize=(8, 6), dpi=150)

            # Create grid for subplots
            gs = fig.add_gridspec(4, 1, hspace=0.3, height_ratios=[3, 1, 3, 1])

            # Main histogram with KDE
            ax1 = fig.add_subplot(gs[0, 0])
            values = data[col].dropna()

            # Histogram
            n, bins, patches = ax1.hist(values, bins=20, density=True, alpha=0.7,
                                        color='steelblue', edgecolor='black', linewidth=0.5)

            # KDE plot
            from scipy import stats
            kde = stats.gaussian_kde(values)
            x_range = np.linspace(values.min(), values.max(), 200)
            ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

            # Add mean and median lines
            mean_val = values.mean()
            median_val = values.median()
            ax1.axvline(mean_val, color='green', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.3f}')
            ax1.axvline(median_val, color='orange', linestyle=':', linewidth=1.5, label=f'Median: {median_val:.3f}')

            ax1.set_ylabel('Density')
            ax1.set_title(f'{display_name} Distribution', fontsize=12, fontweight='bold')
            ax1.legend(loc='upper right', fontsize=8)
            ax1.grid(True, alpha=0.2)

            # Boxplot below first histogram
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


        # Create 2-column layout for the enhanced plots
        cols = st.columns(2)
        plot_count = 0

        for col, display_name in dist_cols:
            if col in results_df.columns and not results_df[col].isna().all():
                with cols[plot_count % 2]:
                    fig = enhanced_distribution_plot(results_df, col, display_name)
                    st.pyplot(fig, width='stretch')
                    plt.close(fig)

                    # Add statistics in a small table below the plot
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

    # Enhanced general statistics with proper naming
    with st.expander("📋 Summary Statistics", expanded=EXPANDED):
        stat_cols = [
            "Kael", "angularity", "surface_roughness",
            "asymmetricity1", "asymmetricity2", "polygonality",
            "elongation", "bulkiness", "surface", "circularity", "sphericity"
        ]
        stat_cols = [c for c in stat_cols if c in results_df.columns]

        # Create statistics table with display names
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

    # Enhanced correlation matrix with triangle=True and automated observations
    with st.expander("🔗 Correlation Matrix", expanded=EXPANDED):
        corr_cols = [
            "Kael", "angularity", "surface_roughness",
            "asymmetricity1", "asymmetricity2", "polygonality",
            "elongation", "bulkiness", "surface", "circularity", "sphericity"
        ]
        corr_cols = [c for c in corr_cols if c in results_df.columns]

        # Rename columns for display
        corr_df = results_df[corr_cols].copy()
        corr_df.rename(columns=display_names, inplace=True)

        corr = corr_df.apply(pd.to_numeric, errors="coerce").corr(method="pearson")

        st.download_button(
            "⬇️ Download correlation matrix (CSV)",
            data=corr.to_csv().encode("utf-8"),
            file_name="correlation_matrix.csv",
            mime="text/csv",
        )

        # Create correlation matrix with triangle=True
        fig_corr = px.imshow(
            corr,
            aspect="auto",
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            text_auto=".2f",
            title="Correlation Matrix (Pearson)",
        )

        # Make it triangular (show only lower triangle)
        mask = np.triu(np.ones_like(corr), k=1).astype(bool)

        # Update the figure to show only lower triangle
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

        # Automatically generate key observations from correlation matrix
        st.markdown("**Key Observations**")

        # Get unique pairs from lower triangle (excluding diagonal)
        observations = []
        corr_values = []

        for i, row in enumerate(corr.index):
            for j, col in enumerate(corr.columns):
                if i > j:  # Lower triangle
                    val = corr.iloc[i, j]
                    if not np.isnan(val):
                        strength = "Very strong" if abs(val) >= 0.8 else \
                            "Strong" if abs(val) >= 0.6 else \
                                "Moderate" if abs(val) >= 0.4 else \
                                    "Weak" if abs(val) >= 0.2 else "Very weak"
                        direction = "positive" if val > 0 else "negative"
                        observations.append(
                            f"- {strength} {direction} correlation between **{row}** and **{col}** ({val:.2f})")
                        corr_values.append(abs(val))

        # Sort observations by absolute correlation strength and show top 8
        sorted_obs = [x for _, x in sorted(zip(corr_values, observations), reverse=True)]

        st.markdown("**Top strongest correlations:**")
        for obs in sorted_obs[:8]:  # Show top 8 strongest correlations
            st.markdown(obs)
        # Add correlation interpretation guide
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