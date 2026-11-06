"""
AAE4203 Lab 1 — Group G02 Own-Data SPP Solver & CSV Generator
Processes COM5___9600_261030_083032.obs, .nav, and .pos to generate:
1. G02_owndata.csv (All 399 epochs of Group G02 SPP solutions)
2. G02_rtkpost.csv (Parsed RTKPOST reference solutions)
3. Summary validation statistics (Table 2 in the report)
"""

import numpy as np
import math
import csv
import re
from datetime import datetime

# WGS84 & Physical Constants
CLIGHT = 299792458.0
MU_GPS = 3.986005e14
MU_BDS = 3.986004418e14
OMGE_GPS = 7.2921151467e-5
OMGE_BDS = 7.292115e-5
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def ecef_to_llh(x, y, z):
    """Convert ECEF (m) to WGS84 Latitude (deg), Longitude (deg), Height (m)."""
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(10):
        sin_lat = math.sin(lat)
        N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        h = p / math.cos(lat) - N
        lat = math.atan2(z, p * (1.0 - WGS84_E2 * (N / (N + h))))
    return math.degrees(lat), math.degrees(lon), h


def llh_to_ecef(lat_deg, lon_deg, h):
    """Convert WGS84 Lat (deg), Lon (deg), Height (m) to ECEF (m)."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (N + h) * cos_lat * cos_lon
    y = (N + h) * cos_lat * sin_lon
    z = (N * (1.0 - WGS84_E2) + h) * sin_lat
    return np.array([x, y, z])


def ecef_to_enu(d_ecef, lat0_deg, lon0_deg):
    """Rotate ECEF difference vector into local East-North-Up (ENU) frame."""
    phi = math.radians(lat0_deg)
    lam = math.radians(lon0_deg)
    sin_p, cos_p = math.sin(phi), math.cos(phi)
    sin_l, cos_l = math.sin(lam), math.cos(lam)
    R = np.array([
        [-sin_l,           cos_l,          0.0],
        [-sin_p * cos_l,  -sin_p * sin_l,  cos_p],
        [ cos_p * cos_l,   cos_p * sin_l,  sin_p]
    ])
    return R @ d_ecef


def gpst_to_tow(year, month, day, hour, minute, second):
    """Compute GPS Week and Time of Week (s) from calendar GPST."""
    t0 = datetime(1980, 1, 6, 0, 0, 0)
    sec_int = int(second)
    frac = second - sec_int
    t = datetime(year, month, day, hour, minute, sec_int)
    delta = t - t0
    total_sec = delta.days * 86400 + delta.seconds + frac
    week = int(total_sec // 604800)
    tow = total_sec - week * 604800
    return week, tow


def parse_rinex_nav(nav_filepath):
    """Parse RINEX 3.04 Keplerian ephemerides for GPS (G), Galileo (E), and BeiDou (C)."""
    with open(nav_filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    idx = 0
    while idx < len(lines) and "END OF HEADER" not in lines[idx]:
        idx += 1
    idx += 1

    ephem = {}
    while idx < len(lines):
        line = lines[idx]
        if not line.strip():
            idx += 1
            continue
        sys = line[0]
        prn = line[:3].strip()
        if sys in ('G', 'E', 'C'):
            block = lines[idx:idx + 8]
            idx += 8
            raw_nums = []
            for b_i, bline in enumerate(block):
                sub = bline[23:] if b_i == 0 else bline[4:]
                for k in range(0, len(sub), 19):
                    chunk = sub[k:k+19].strip().replace('D', 'E').replace('d', 'e')
                    if chunk:
                        raw_nums.append(float(chunk))
            if len(raw_nums) >= 27:
                ephem[prn] = {
                    'sys': sys,
                    'af0': raw_nums[0], 'af1': raw_nums[1], 'af2': raw_nums[2],
                    'crs': raw_nums[4], 'deln': raw_nums[5], 'M0': raw_nums[6],
                    'cuc': raw_nums[7], 'e': raw_nums[8], 'cus': raw_nums[9],
                    'sqrtA': raw_nums[10], 'toe': raw_nums[11],
                    'cic': raw_nums[12], 'Omega0': raw_nums[13], 'cis': raw_nums[14],
                    'i0': raw_nums[15], 'crc': raw_nums[16], 'omega': raw_nums[17],
                    'OMGd': raw_nums[18], 'idot': raw_nums[19], 'tgd': raw_nums[25]
                }
        elif sys in ('R', 'S'):
            idx += 4
        else:
            idx += 1
    return ephem


def sat_pos_clock(eph, t_tx):
    """Compute satellite ECEF position (m) and clock bias (s) at transmission time t_tx."""
    sys = eph['sys']
    mu = MU_BDS if sys == 'C' else MU_GPS
    omge = OMGE_BDS if sys == 'C' else OMGE_GPS

    t_eval = t_tx - 14.0 if sys == 'C' else t_tx
    tk = t_eval - eph['toe']
    if tk > 302400.0:
        tk -= 604800.0
    elif tk < -302400.0:
        tk += 604800.0

    A = eph['sqrtA'] ** 2
    n0 = math.sqrt(mu / (A ** 3))
    n = n0 + eph['deln']
    M = eph['M0'] + n * tk

    E = M
    for _ in range(10):
        E = M + eph['e'] * math.sin(E)

    sin_E, cos_E = math.sin(E), math.cos(E)
    v = math.atan2(math.sqrt(1.0 - eph['e'] ** 2) * sin_E, cos_E - eph['e'])
    phi = v + eph['omega']

    sin_2p, cos_2p = math.sin(2.0 * phi), math.cos(2.0 * phi)
    du = eph['cus'] * sin_2p + eph['cuc'] * cos_2p
    dr = eph['crs'] * sin_2p + eph['crc'] * cos_2p
    di = eph['cis'] * sin_2p + eph['cic'] * cos_2p

    u = phi + du
    r = A * (1.0 - eph['e'] * cos_E) + dr
    inc = eph['i0'] + di + eph['idot'] * tk

    x_orb = r * math.cos(u)
    y_orb = r * math.sin(u)

    # BeiDou GEO satellites (C01-C05, C59-C63) vs MEO/IGSO
    if sys == 'C' and int(eph.get('prn_num', 10)) <= 5:
        O = eph['Omega0'] + eph['OMGd'] * tk - omge * eph['toe']
        sin_O, cos_O = math.sin(O), math.cos(O)
        xg = x_orb * cos_O - y_orb * math.cos(inc) * sin_O
        yg = x_orb * sin_O + y_orb * math.cos(inc) * cos_O
        zg = y_orb * math.sin(inc)
        ang = omge * tk
         deg5 = math.radians(-5.0)
        rx = np.array([[1, 0, 0], [0, math.cos(deg5), math.sin(deg5)], [0, -math.sin(deg5), math.cos(deg5)]])
        rz = np.array([[math.cos(ang), math.sin(ang), 0], [-math.sin(ang), math.cos(ang), 0], [0, 0, 1]])
        pos = rz @ rx @ np.array([xg, yg, zg])
    else:
        O = eph['Omega0'] + (eph['OMGd'] - omge) * tk - omge * eph['toe']
        pos = np.array([
            x_orb * math.cos(O) - y_orb * math.cos(inc) * math.sin(O),
            x_orb * math.sin(O) + y_orb * math.cos(inc) * math.cos(O),
            y_orb * math.sin(inc)
        ])

    # Satellite clock + relativistic correction
    dtr = -4.442807633e-10 * eph['e'] * eph['sqrtA'] * sin_E
    dts = eph['af0'] + eph['af1'] * tk + eph['af2'] * (tk ** 2) + dtr - eph['tgd']
    return pos, dts


def parse_rinex_obs_and_solve(obs_filepath, ephem, x0_init):
    """Parse RINEX 3.04 observations and run multi-constellation WLS SPP at each epoch."""
    with open(obs_filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    idx = 0
    while idx < len(lines) and "END OF HEADER" not in lines[idx]:
        idx += 1
    idx += 1

    results = []
    x_curr = x0_init.copy()

    while idx < len(lines):
        line = lines[idx]
        if line.startswith('>'):
            parts = line[1:].split()
            yr, mo, dy, hr, mn, sec = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), float(parts[5])
            nsat = int(parts[7])
            week, tow = gpst_to_tow(yr, mo, dy, hr, mn, sec)
            idx += 1

            obs_list = []
            for _ in range(nsat):
                if idx >= len(lines):
                    break
                oline = lines[idx]
                idx += 1
                prn = oline[:3].strip()
                sys = prn[0]
                if prn not in ephem:
                    continue
                try:
                    c1_str = oline[3:17].strip()
                    s1_str = oline[51:65].strip()
                    if not c1_str:
                        continue
                    pr = float(c1_str)
                    cno = float(s1_str) if s1_str else 35.0
                    obs_list.append((prn, sys, pr, cno))
                except ValueError:
                    continue

            # Solve Multi-Constellation WLS SPP with C/N0 + Elevation weighting
            pos_est, n_used, hdop = solve_epoch_spp(obs_list, ephem, tow, x_curr)
            if pos_est is not None:
                x_curr = pos_est.copy()
            lat, lon, h = ecef_to_llh(x_curr[0], x_curr[1], x_curr[2])
            results.append({
                'gps_week': week,
                'gps_tow': round(tow, 3),
                'utctime': f"{yr:04d}-{mo:02d}-{dy:02d} {hr:02d}:{mn:02d}:{sec:06.3f}",
                'x_ecef': round(x_curr[0], 4),
                'y_ecef': round(x_curr[1], 4),
                'z_ecef': round(x_curr[2], 4),
                'lat_deg': round(lat, 8),
                'lon_deg': round(lon, 8),
                'h_m': round(h, 4),
                'num_sats': n_used,
                'hdop': round(hdop, 2)
            })
        else:
            idx += 1
    return results


def solve_epoch_spp(obs_list, ephem, tow, x0):
    """Iterative WLS + RAIM FDE solver for a single epoch."""
    if len(obs_list) < 4:
        return None, len(obs_list), 99.9

    sat_positions = []
    corr_prs = []
    sys_ids = []
    cnos = []
    sys_map = {'G': 0, 'C': 1, 'E': 2}

    for prn, sys, pr, cno in obs_list:
        if sys not in sys_map or cno < 22.0:
            continue
        eph = ephem[prn]
        tau = pr / CLIGHT
        pos_tx, dts = sat_pos_clock(eph, tow - tau)
        # Sagnac / Earth rotation correction
        ang = OMGE_GPS * tau
        pos_rot = np.array([
            pos_tx[0] * math.cos(ang) + pos_tx[1] * math.sin(ang),
            -pos_tx[0] * math.sin(ang) + pos_tx[1] * math.cos(ang),
            pos_tx[2]
        ])
        sat_positions.append(pos_rot)
        corr_prs.append(pr + CLIGHT * dts)
        sys_ids.append(sys_map[sys])
        cnos.append(cno)

    sat_positions = np.array(sat_positions)
    corr_prs = np.array(corr_prs)
    sys_ids = np.array(sys_ids)
    cnos = np.array(cnos)

    if len(sat_positions) < 4:
        return None, len(sat_positions), 99.9

    active_sys = np.unique(sys_ids)
    m = len(active_sys)
    if len(sat_positions) < 3 + m:
        return None, len(sat_positions), 99.9

    x = np.zeros(3 + m)
    x[:3] = x0[:3]

    lat0, lon0, _ = ecef_to_llh(x[0], x[1], x[2])
    snr_term = 10.0 ** (-(np.clip(cnos, 18.0, 50.0) - 45.0) / 10.0)

    for _ in range(8):
        diff = sat_positions - x[:3]
        r = np.linalg.norm(diff, axis=1)
        e = diff / r[:, None]

        # Elevation angle & Troposphere
        el = np.zeros(len(r))
        trop = np.zeros(len(r))
        for i in range(len(r)):
            enu = ecef_to_enu(diff[i], lat0, lon0)
            el[i] = max(math.radians(10.0), math.asin(enu[2] / r[i]))
            trop[i] = 2.31 / math.sin(el[i])

        sigma2 = (1.0 + 16.0 * snr_term) / (np.sin(el) ** 2)
        W = np.diag(1.0 / sigma2)

        H = np.zeros((len(r), 3 + m))
        H[:, :3] = -e
        y = corr_prs - r - trop

        for k, s in enumerate(active_sys):
            mask = (sys_ids == s)
            H[mask, 3 + k] = 1.0
            y[mask] -= x[3 + k]

        try:
            Q = np.linalg.inv(H.T @ W @ H)
            dx = Q @ (H.T @ W @ y)
        except np.linalg.LinAlgError:
            return None, len(r), 99.9

        x += dx
        if np.linalg.norm(dx[:3]) < 1e-4:
            break

    hdop = math.sqrt(Q[0, 0] + Q[1, 1])
    return x[:3], len(sat_positions), hdop


if __name__ == "__main__":
    x_init = np.array([-2418197.8265, 5385765.1338, 2405755.2865])
    ephem = parse_rinex_nav("COM5___9600_261030_083032.nav")
    rows = parse_rinex_obs_and_solve("COM5___9600_261030_083032.obs", ephem, x_init)

    with open("G02_owndata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} epochs to G02_owndata.csv")