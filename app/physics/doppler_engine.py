"""
Acoustic Doppler physics engine for aircraft geolocator.
Calculates 3D slant distances, sound propagation delays, radial velocities,
and theoretical observed Doppler frequency curves.
"""

import math
from typing import Tuple, Union
import numpy as np


class DopplerEngine:
    def __init__(self, speed_of_sound: float = 343.2, earth_radius: float = 6371000.0):
        """
        :param speed_of_sound: Speed of sound in m/s (default 343.2 m/s at 20°C)
        :param earth_radius: Mean Earth radius in meters (default 6,371,000 m)
        """
        self.c = float(speed_of_sound)
        self.R = float(earth_radius)

    @staticmethod
    def calculate_speed_of_sound(temp_celsius: float = 20.0, relative_humidity: float = 0.5,
                                 pressure_pa: float = 101325.0, co2_mole_fraction: float = 0.0004) -> float:
        """
        Speed of sound in humid air (Cramer 1993, J. Acoust. Soc. Am. 93, 2510).

        :param temp_celsius: air temperature [degC]
        :param relative_humidity: relative humidity as a fraction 0..1 (values > 1 are read as percent)
        :param pressure_pa: static pressure [Pa]
        :param co2_mole_fraction: CO2 mole fraction (0.0004 ~ present-day air)

        Water-vapour mole fraction uses the Davis (1992) saturation vapour
        pressure and enhancement factor, as in Cramer's paper.  Accurate to
        ~0.1 m/s for 0-30 degC; outside that range it remains a smooth
        approximation.
        """
        h = float(relative_humidity)
        if h > 1.0:
            h /= 100.0
        h = min(max(h, 0.0), 1.0)
        t = float(temp_celsius)
        T = t + 273.15
        p = float(pressure_pa)
        p_sv = math.exp(1.2811805e-5 * T * T - 1.9509874e-2 * T + 34.04926034 - 6.3536311e3 / T)
        f_enh = 1.00062 + 3.14e-8 * p + 5.6e-7 * t * t
        x_w = h * f_enh * p_sv / p
        x_c = float(co2_mole_fraction)
        a = (331.5024, 0.603055, -0.000528, 51.471935, 0.1495874, -0.000782, -1.82e-7, 3.73e-8, -2.93e-10,
             -85.20931, -0.228525, 5.91e-5, -2.835149, -2.15e-13, 29.179762, 0.000486)
        return (a[0] + a[1] * t + a[2] * t * t
                + (a[3] + a[4] * t + a[5] * t * t) * x_w
                + (a[6] + a[7] * t + a[8] * t * t) * p
                + (a[9] + a[10] * t + a[11] * t * t) * x_c
                + a[12] * x_w * x_w + a[13] * p * p + a[14] * x_c * x_c + a[15] * x_w * p * x_c)

    def geodetic_to_cartesian(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        alt: np.ndarray,
        lat_ref: float,
        lon_ref: float,
        alt_ref: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert WGS84 (lat, lon, alt_meters) to local tangent Cartesian coordinates (x_east, y_north, z_up) in meters.
        """
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        lat_ref_rad = math.radians(lat_ref)
        lon_ref_rad = math.radians(lon_ref)

        lat_mid = (lat_rad + lat_ref_rad) / 2.0
        x = self.R * (lon_rad - lon_ref_rad) * np.cos(lat_mid)
        y = self.R * (lat_rad - lat_ref_rad)
        z = alt - alt_ref
        return x, y, z

    def cartesian_to_geodetic(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        lat_ref: float,
        lon_ref: float,
        alt_ref: float = 0.0,
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Convert local Cartesian coordinates (meters) back to WGS84 (lat, lon, alt).
        """
        lat_ref_rad = math.radians(lat_ref)
        d_lat = y / self.R
        lat_rad = lat_ref_rad + d_lat
        lat_mid = (lat_rad + lat_ref_rad) / 2.0
        d_lon = x / (self.R * np.cos(lat_mid))
        lon_rad = math.radians(lon_ref) + d_lon

        lat_deg = np.degrees(lat_rad)
        lon_deg = np.degrees(lon_rad)
        alt = z + alt_ref
        return lat_deg, lon_deg, alt

    def compute_radial_velocity(
        self,
        plane_pos: np.ndarray,
        plane_vel: np.ndarray,
        obs_pos: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute slant distance and radial velocity of the aircraft relative to observer.
        
        :param plane_pos: (N, 3) Cartesian positions of plane [x, y, z] in meters
        :param plane_vel: (N, 3) Cartesian velocity vectors of plane [vx, vy, vz] in m/s
        :param obs_pos: (3,) Cartesian position of candidate observer [x, y, z] in meters
        :return: (slant_distances, radial_velocities_towards_observer)
        """
        # Vector from observer to plane: r_vec = plane_pos - obs_pos
        r_vec = plane_pos - obs_pos
        # 3D Slant distance (hypotenuse)
        d = np.linalg.norm(r_vec, axis=-1)
        d = np.maximum(d, 1.0)  # Avoid division by zero

        # Radial velocity towards the observer:
        # v_r = - (r_vec . v_plane) / d
        # When moving closer (r_vec . v_plane < 0), v_r > 0
        v_r = -np.sum(r_vec * plane_vel, axis=-1) / d
        return d, v_r

    def compute_doppler_curve(
        self,
        t_emit: np.ndarray,
        plane_pos: np.ndarray,
        plane_vel: np.ndarray,
        obs_pos: np.ndarray,
        f0: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute theoretical observed Doppler curve accounting for acoustic travel time delay.

        Doppler Equation:
            f = f_0 * (c / (c - v_r))
        Acoustic propagation delay:
            t_obs = t_emit + d / c

        :return: (t_obs, f_obs)
        """
        d, v_r = self.compute_radial_velocity(plane_pos, plane_vel, obs_pos)

        # Ensure Mach limit check for numerical safety (subsonic acoustic modeling)
        # v_r clamped below speed of sound c
        v_r_safe = np.clip(v_r, -0.95 * self.c, 0.95 * self.c)

        # Classical Doppler equation for moving emitter relative to stationary observer:
        # f = f_0 * (c / (c - v_r))
        f_obs = f0 * (self.c / (self.c - v_r_safe))

        # Sound propagation delay: sound emitted at t_emit arrives at observer at t_obs
        t_obs = t_emit + (d / self.c)

        return t_obs, f_obs

    def evaluate_model_at_times(
        self,
        t_query: np.ndarray,
        t_emit: np.ndarray,
        plane_pos: np.ndarray,
        plane_vel: np.ndarray,
        obs_pos: np.ndarray,
        f0: float,
    ) -> np.ndarray:
        """
        Evaluate modeled frequency at specified observer timestamps t_query.
        Uses monotonic interpolation from the delay-mapped curve.
        """
        t_obs, f_obs = self.compute_doppler_curve(t_emit, plane_pos, plane_vel, obs_pos, f0)
        # Sort in case of any slight numerical jitter in t_obs
        sort_idx = np.argsort(t_obs)
        t_obs_sorted = t_obs[sort_idx]
        f_obs_sorted = f_obs[sort_idx]

        f_interp = np.interp(t_query, t_obs_sorted, f_obs_sorted)
        return f_interp
