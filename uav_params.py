"""Parameters for a small fixed-wing surveillance UAV (~20 kg class).

Think of a scaled-down Predator/Reaper type: high wing, pusher propeller,
built for endurance ISR (intelligence, surveillance, reconnaissance) work.
The numbers below are plausible engineering approximations for a 20 kg
electric UAV -- good enough for a believable desktop simulator and for
learning how each parameter shapes handling. They are NOT manufacturer data.

Basis (documented so you can sanity-check or retune):
  - Mass 20 kg, wing area 2.5 m^2, span 4.2 m -> wing loading ~78 N/m^2.
    (A light wing loading like this is typical of small surveillance UAVs;
    it keeps the stall speed low, around 9-10 m/s.)
  - Stall speed estimate: Vs = sqrt(2*W / (rho*S*CLmax)) with
    CLmax ~= CL0 + CLa*alpha_crit = 0.3 + 4.8*0.26 ~= 1.55
    -> Vs ~= sqrt(392 / (1.225*2.5*1.55)) ~= 9.1 m/s (~18 kts).
  - Cruise ~22-25 m/s. Thrust: 95 N static gives T/W ~0.48, plenty for
    a 3-4 m/s climb at 20 kg.
  - Inertias scaled from light-aircraft textbook values by the mass and
    size ratio (a 20 kg / 3 m airframe is roughly 1/370th the inertia
    of a 1043 kg / 8 m one).
  - Stability/control derivatives are typical small-UAV values: positively
    stable in pitch, mild dihedral, adequate rudder authority.

Sign conventions (same as the companion airplane sim -- body frame x
forward, y right, z up, right-handed):
  - elevator +1 = pull / nose up.  Cmde < 0 gives the nose-up moment.
  - aileron +1 = roll right.  Positive p (about +x) is roll LEFT, so
    Clda < 0 rolls right for positive input.
  - rudder +1 = nose right (positive r about +z).
  - Pitch moment about +y: positive = nose DOWN, so Cma > 0 is stable.
"""

from dataclasses import dataclass


@dataclass
class UAVParams:
    """Complete parameter set for the simulated surveillance UAV."""

    name: str = "20-kg fixed-wing surveillance UAV (approx)"

    # --- Mass & inertia -------------------------------------------------
    mass: float = 20.0            # kg
    Ixx: float = 3.5              # kg m^2, roll inertia
    Iyy: float = 5.0              # kg m^2, pitch inertia
    Izz: float = 7.5              # kg m^2, yaw inertia

    # --- Wing geometry --------------------------------------------------
    wing_area: float = 2.5        # m^2
    wing_span: float = 4.2        # m
    wing_chord: float = 0.595     # m, mean aerodynamic chord (area/span)

    # --- Lift / drag ----------------------------------------------------
    CL0: float = 0.30             # lift coeff at alpha = 0 (cambered wing)
    CL_alpha: float = 4.8         # per rad, lift-curve slope
    alpha_crit: float = 0.26      # rad (~15 deg), stall angle
    CD0: float = 0.035            # parasite drag coefficient
    induced_k: float = 0.060      # induced drag factor 1/(pi*AR*e)
    CY_beta: float = -0.6         # per rad, sideforce due to sideslip

    # --- Pitch ----------------------------------------------------------
    Cm0: float = 0.02             # pitching moment at alpha = 0
    Cma: float = 0.8              # per rad, pitch stiffness (stable > 0)
    Cmq: float = -10.0            # per rad, pitch damping
    Cmde: float = -1.1            # per rad, elevator effectiveness

    # --- Roll -----------------------------------------------------------
    Clb: float = 0.08             # per rad, dihedral effect
    Clp: float = -0.40            # per rad, roll damping
    Clda: float = -0.16           # per rad, aileron effectiveness
    Cldr: float = -0.02           # per rad, roll due to rudder

    # --- Yaw ------------------------------------------------------------
    Cnb: float = 0.11             # per rad, weathervane stability
    Cnr: float = -0.14            # per rad, yaw damping
    Cnda: float = -0.02           # per rad, adverse yaw from ailerons
    Cndr: float = 0.09            # per rad, rudder effectiveness

    # --- Propulsion (electric pusher prop) -------------------------------
    # Same thrust shape as the airplane sim: the lesser of a static-thrust
    # cap (dominates at low speed) and available power / airspeed.
    thrust_static: float = 95.0   # N at full throttle, zero airspeed
    prop_power: float = 4000.0    # W effective propulsive power

    # --- Control limits -------------------------------------------------
    elevator_limit: float = 0.40  # rad of surface deflection
    aileron_limit: float = 0.35   # rad
    rudder_limit: float = 0.30    # rad
    trim_authority: float = 0.6   # trim adds up to +/-60% of elevator travel

    # --- Performance reference (for autopilot / docs) --------------------
    cruise_speed: float = 22.0    # m/s
    stall_speed: float = 9.5      # m/s (approx, from the Vs estimate above)
    rotate_speed: float = 12.0    # m/s, liftoff target on takeoff


def SurveillanceUAV() -> UAVParams:
    """Factory returning the default fixed-wing UAV parameter set."""
    return UAVParams()
