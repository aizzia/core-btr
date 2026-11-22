import json
import os
import shutil
import h5py
import numpy as np

OUT_DIR = "mock_duke_h5"
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(108)

# Realistic DAX / ARTIQ trapped-ion calibration regimes sharing common DAX experiment classes & beam names
# so text RAG sees identical surface words ('SpectroscopyScan', '397nm_doppler', '729nm_carrier')
REGIMES = (
    ["healthy_rabi_flop"] * 85
    + ["ramsey_60hz_bfield_drift"] * 85
    + ["midscan_ion_loss_dark"] * 85
    + ["scarce_aom_thermal_droop"] * 28   # Scarce in index split
    + ["novel_laser_unlock_spike"] * 12   # Held-out only -> must trigger EMPTY_BELOW_CUTOFF
)

EXP_CLASSES = ["DAXSpectroscopyScan", "DAXCalibrationLoop", "DAXParityCheck"]
BEAMS = ["729nm_quadrupole", "397nm_doppler", "854nm_repump"]

for idx, regime in enumerate(REGIMES):
    rid = 50000 + idx
    n_steps = 40
    t = np.linspace(0.0, 1.0, n_steps)
    tier = 1.0 if idx % 2 == 0 else 1.8

    exp_cls = EXP_CLASSES[idx % 3]
    beam = BEAMS[(idx // 2) % 3]

    with h5py.File(f"{OUT_DIR}/000{rid}-artiq.h5", "w") as f:
        f["rid"] = rid
        f["start_time"] = 1763500000.0 + idx * 420.0 + np.random.randint(0, 120)
        f["expid"] = json.dumps({
            "class_name": exp_cls,
            "beam_channel": beam,
            "regime_tag": regime,
        }).encode()
        ds = f.create_group("datasets")

        if regime == "healthy_rabi_flop":
            # Clean damped Rabi oscillation; spectator diagnostic channel unprobed (NaN) on odd steps
            pmt = np.round((22.0 * tier) * (np.sin(3.5 * np.pi * t) ** 2) * np.exp(-0.35 * t) + 2.0)
            spec_diag = np.where(np.arange(n_steps) % 2 == 1, np.nan, np.round(18.0 + np.random.normal(0, 1.2, n_steps)))
            offsets = np.cumsum(np.full(n_steps, 0.04) + np.random.uniform(0, 0.005, n_steps))
            atten = 13.5

        elif regime == "ramsey_60hz_bfield_drift":
            # 60 Hz AC line magnetic phase modulation causing Ramsey contrast collapse & revival + bursty waits
            pmt = np.round((16.0 * tier) * (0.5 + 0.48 * np.cos(7.0 * np.pi * t) * np.exp(-2.2 * np.sin(np.pi * t) ** 2)) + 1.0)
            spec_diag = np.where(np.arange(n_steps) % 3 == 0, np.nan, np.round(14.0 - 8.0 * t + np.random.normal(0, 1.0, n_steps)))
            gaps = np.where(np.arange(n_steps) % 4 == 0, 0.45, 0.0166)  # 16.6ms (60Hz period) w/ re-lock pauses
            offsets = np.cumsum(gaps)
            atten = 14.0

        elif regime == "midscan_ion_loss_dark":
            # Ion starts bright, then collision/heating hops it dark -> EXPLICIT 0.0 photon counts (NOT NaN!)
            hop_idx = np.random.randint(16, 24)
            pmt = np.round((20.0 * tier) + np.random.normal(0, 1.8, n_steps)).clip(3.0, 45.0)
            pmt[hop_idx:] = 0.0
            spec_diag = np.zeros(n_steps, dtype=float)
            spec_diag[:hop_idx] = np.round(19.0 + np.random.normal(0, 1.0, hop_idx))
            spec_diag[hop_idx:] = 0.0
            offsets = np.cumsum(np.full(n_steps, 0.08) + np.random.uniform(0, 0.02, n_steps))
            atten = 13.5

        elif regime == "scarce_aom_thermal_droop":
            # Rapid micro-spaced shots causing AOM crystal thermal droop + alternating 0.0 dark calibration checks
            pmt = np.where(np.arange(n_steps) % 2 == 0, 0.0, np.round(26.0 * np.exp(-2.0 * t) + 1.0))
            spec_diag = np.where(np.arange(n_steps) % 2 == 0, np.nan, 0.0)
            offsets = np.cumsum(np.random.uniform(0.001, 0.003, n_steps))
            atten = 19.5

        else:  # novel_laser_unlock_spike -> saturation PMT spikes & multi-minute timeout stalls
            pmt = np.round(np.random.uniform(4500.0, 8000.0, n_steps))
            spec_diag = np.round(np.random.uniform(3000.0, 6000.0, n_steps))
            offsets = np.cumsum(np.random.uniform(90.0, 180.0, n_steps))
            atten = 2.0

        ds["pmt_counts"] = pmt
        ds["spectator_counts"] = spec_diag
        ds["shot_offsets_s"] = offsets
        ds["aom_atten_db"] = atten

print(f"[Mock ARTIQ Generator] Wrote {len(REGIMES)} native .h5 experiment files to ./{OUT_DIR}")
