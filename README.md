# RTL-SDR H-Line Scanner + Distributed Synthesis Data Collector

A Raspberry Pi + DRV8833 stepper + RTL-SDR system for two things:

1. **H-line viewing** - point at a target, watch PSD / hydrogen-line power live.
2. **Synthesis-imaging data collection** - long, GPS-timestamped, precisely-pointed
   raw IQ captures designed so multiple independent stations' recordings could
   later be cross-correlated for Earth-rotation ("rotational") synthesis. This
   codebase does **not** do that correlation/imaging step - it only makes sure
   the collected data has what a correlator would need.

## Why no IMU / magnetometer

Two deliberate choices, per your hardware constraints:

- **Azimuth steps/degree** is auto-calibrated by counting steps between two
  triggers of a hall sensor mounted on the azimuth platform (one trigger per
  full 360&deg; rotation). `stepper_control.DualAxisMount.calibrate_azimuth_steps_per_degree()`.
- **Absolute pointing accuracy** (both axes) is calibrated by beam-peaking
  against known bright radio sources (Cas A, Cyg A, the Crab, Galactic Center,
  or the Sun) computed via `astropy` from your GPS position/time, then doing a
  small grid search with the SDR to find where the signal actually peaks.
  `peak_calibrator.peak_calibrate()`. This sidesteps a magnetometer entirely,
  which is good, because a magnetometer mounted a few inches from a stepper
  motor's field is not going to give you a usable compass reading anyway.

Elevation steps/degree still needs either a one-time manual measurement
(`mount.set_elevation_steps_per_degree(value)`) or two peak-calibration runs at
different elevations to fit from - see the notes in `peak_calibrator.py`. There's
no hall-sensor shortcut for elevation because most mounts don't do a full 360&deg;
sweep on that axis.

## File map

| File | Role |
|---|---|
| `config.py` | Wiring, site fallbacks, SDR settings - edit this first |
| `gps_time.py` | GPS position + GPS-derived UTC time (spatial + temporal ground truth) |
| `stepper_control.py` | DRV8833 dual-axis mount: homing, az auto-cal, absolute Az/El positioning |
| `coordinate_transforms.py` | RA/Dec &harr; Alt/Az via astropy, bright-source catalog |
| `sdr_capture.py` | PSD measurement + H-line detection + raw IQ capture |
| `peak_calibrator.py` | Beam-peaking pointing calibration (no IMU needed) |
| `data_logger.py` | Self-describing capture storage (station ID, GPS, pointing, data) |
| `observation_modes.py` | `h_line_stare`, `raster_scan`, `synthesis_track_session` |
| `main.py` | CLI: runs startup sequence, then your chosen mode |

Everything falls back to a mock/simulated mode when RTL-SDR, RPi.GPIO, or a real
GPS fix isn't available, so you can develop and test off-Pi.

## Wiring assumptions (edit `config.py` if yours differs)

- DRV8833 azimuth: GPIO 17/18. Elevation: GPIO 22/23.
- Azimuth hall sensor: GPIO 5, mounted so it triggers once per full 360&deg;
  rotation of the platform (not the motor shaft, unless they're 1:1).
- Elevation hall sensor: GPIO 6, a single home/reference position (e.g. horizon stop).
- GPS: serial, default `/dev/ttyS0` @ 9600 baud.
- RTL-SDR: default USB, tuned to 1.42040575 GHz (HI rest frequency), 2.048 Msps.

## Quick start

```bash
pip install -r requirements.txt --break-system-packages   # on-device, drop the flag if using a venv

# Just run startup (home, calibrate az, get GPS fix, peak-cal) and stop:
python3 main.py --mode home-only

# H-line viewing mode - watch a target for 10 minutes:
python3 main.py --mode stare --ra 83.633 --dec 22.0145 --duration 600 --interval 5

# Raster scan a patch of sky (single-dish sky map data collection):
python3 main.py --mode raster --ra 266.4 --dec -29.0 --half-width 6 --step 1

# Synthesis-collection session - 1 hour of GPS-tagged raw IQ chunks:
python3 main.py --mode synthesis --ra 299.868 --dec 40.734 --duration 3600 --chunk 2
```

Data lands in `./observations/<session_id>/`, one JSON metadata sidecar plus
`.npy` data file(s) per capture. See `data_logger.py`'s docstring for the exact
layout.

## Honest limitations - read before you trust the synthesis data

- **Timing**: GPS-derived UTC from NMEA sentences is good to roughly tens of
  milliseconds, not the microsecond-level timing a real VLBI-style correlator
  wants. That's fine for long-integration correlation of steady continuum
  sources across an Earth-rotation synthesis session; it is **not** adequate
  for fine fringe-rate tracking. A GPS-disciplined oscillator (GPSDO) feeding
  a PPS-tagged sample clock is the real upgrade path if you want to go further,
  and it's a hardware change, not something software can paper over.
- **Elevation calibration** is coarser than azimuth's, because there's no
  automatic full-rotation reference for it. Budget real time for the two-point
  peak-calibration fit, or a one-time manual protractor measurement.
- **Baseline vectors** (the relative positions between stations) are only as
  good as each station's GPS fix. Consumer GPS is meter-level; for real
  synthesis-imaging quality you'd eventually want surveyed or differential
  (RTK) positions, especially for short baselines where a few meters of error
  is a large fraction of the baseline itself.
- **No correlation/imaging code is included on purpose** (per your ask) - this
  system's job ends at producing clean, well-tagged IQ/PSD data. The natural
  next step, whenever you're ready, is a separate offline pipeline that reads
  multiple stations' `session_metadata.json` + capture files, aligns them by
  `utc_time`, and cross-correlates.
