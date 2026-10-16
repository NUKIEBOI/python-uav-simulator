"""Electric propulsion battery model (6S LiPo).

HEADLESS: pure Python, no NumPy needed. It tracks state of charge from the
throttle setting, sags the terminal voltage under load, and raises low and
critical warnings. The flight code (simulator / autopilot) decides what to do
about them -- typically a warning banner, then an emergency auto-land.

Model (simple but honest):
  * Open-circuit voltage falls linearly with state of charge: 25.2 V full
    (4.2 V/cell) down to 18.0 V empty (3.0 V/cell).
  * Terminal voltage = open-circuit voltage - current * internal resistance.
  * Current draw grows a little faster than linearly with throttle
    (propeller loading), plus a small avionics baseline.
  * Charge integrates current over time: d(SoC) = -I*dt / capacity.
"""

LOW_SOC = 0.25       # below this: "LOW BATTERY" warning
CRITICAL_SOC = 0.12  # below this: "CRITICAL" -> auto-land


class Battery:
    def __init__(self, capacity_ah=20.0, cells=6, r_internal=0.02):
        self.capacity_ah = float(capacity_ah)
        self.cells = int(cells)
        self.r_internal = float(r_internal)   # ohms, whole pack
        self.v_full = 4.2 * self.cells        # 25.2 V for 6S
        self.v_empty = 3.0 * self.cells       # 18.0 V for 6S
        self.reset()

    def reset(self):
        self.soc = 1.0            # state of charge, 1 = full
        self.voltage = self.v_full
        self.current_a = 0.0
        self.low_battery = False
        self.critical_battery = False

    def step(self, dt: float, throttle: float):
        """Advance `dt` seconds at the given throttle (0..1)."""
        dt = max(0.0, min(float(dt), 1.0))
        thr = max(0.0, min(float(throttle), 1.0))

        # Avionics baseline + motor draw (superlinear in throttle, like a
        # real prop: power ~ rpm^3, current follows).
        self.current_a = 2.0 + 58.0 * (thr ** 1.2)

        # Integrate charge out of the pack.
        self.soc -= self.current_a * dt / (self.capacity_ah * 3600.0)
        self.soc = max(0.0, min(1.0, self.soc))

        # Terminal voltage: open-circuit minus sag under load.
        v_oc = self.v_empty + (self.v_full - self.v_empty) * self.soc
        self.voltage = max(0.0, v_oc - self.current_a * self.r_internal)

        self.low_battery = self.soc < LOW_SOC
        self.critical_battery = self.soc < CRITICAL_SOC
        return self.voltage

    @property
    def percent(self) -> float:
        return 100.0 * self.soc

    def __repr__(self):
        return (f"Battery(soc={self.soc:.3f}, V={self.voltage:.1f}, "
                f"I={self.current_a:.1f}A)")
