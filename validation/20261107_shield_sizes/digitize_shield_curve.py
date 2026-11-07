"""Read the MUMETALL trace from VAC's published static log-log chart.

This is chart digitization, not a manufacturer-supplied tabulated measurement.
Pixel anchors select the labelled trace; no fit to simulation output is used.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parent
image = np.asarray(Image.open(ROOT / 'VAC_static_chart_0.png').convert('L'))
ink = image > 128  # The extracted 1-bit PDF image has reverse polarity.
# Grid centers read from the original 7216 x 6352 image. H axis in A/cm.
xgrid = np.array([493.5, 2156., 3836.5, 5514.5, 7178.])
ygrid = np.array([5853.5, 4179.5, 2506., 834.])
# Manual branch anchors are pixel positions on the second, MUMETALL, trace.
anchor_x = np.array([550,1000,1500,2000,2500,3000,3500,4000,4500,5000,6000,7000])
anchor_y = np.array([4640,4127,3334,2226,1445,1228,1192,1173,1152,1134,1114,1104])
guide = PchipInterpolator(anchor_x, anchor_y, extrapolate=True)
rows = []
for x in np.arange(535, 7121, 100):
    if np.min(abs(xgrid-x)) < 30:
        continue  # A vertical grid line merges with every curve.
    expected = float(guide(x))
    candidates = np.flatnonzero(ink[:, x])
    runs = np.split(candidates, np.flatnonzero(np.diff(candidates) > 1) + 1)
    runs = [r for r in runs if len(r) >= 15 and abs(float(r.mean())-expected) < 180]
    if not runs:
        raise RuntimeError(f'No unambiguous curve ink near pixel {(x, expected)}')
    selected = min(runs, key=lambda r: abs(float(r.mean())-expected))
    # On the plateau, intersections/overlap cannot be resolved to a centerline.
    # Omit ambiguous readings rather than fabricate samples through an overlap.
    if x > 3000 and len(selected) > 25:
        continue
    y = float(selected.mean())
    # Grid slopes in the scanned image are less than ~10 px across the chart;
    # the stated reading uncertainty below covers this as well as line thickness.
    H = 10 ** np.interp(float(x), xgrid, [-1,0,1,2,3])
    B = 10 ** np.interp(y, ygrid[::-1], [0,-1,-2,-3])
    rows.append([int(x), y, H, B])
raw = np.asarray(rows)
assert np.all(np.diff(raw[:, 2]) > 0)
if np.any(np.diff(raw[:, 3]) < -1e-3):
    print('Backsteps:', [(raw[i].tolist(), raw[i+1].tolist()) for i in np.flatnonzero(np.diff(raw[:,3]) < -1e-3)], flush=True)
    np.savetxt(ROOT/'shield_curve_reading_diagnostic.csv',raw,delimiter=',')
    raise RuntimeError('Digitized trace moves backwards; inspect the branch selection')
# Discard tiny scan/line-thickness backsteps without changing retained knot values.
accepted = [0]
for i in range(1, len(raw)):
    if raw[i, 3] > raw[accepted[-1], 3]:
        accepted.append(i)
knots = raw[accepted, 2:]
mu0 = 4e-7*np.pi
assert np.all(np.diff(knots[:, 1]/mu0-knots[:, 0]) > 0)
# Origin and constant-M high-field tail are explicit modelling assumptions.
curve = np.vstack([[0,0], knots, [200000, knots[-1,1]+mu0*(200000-knots[-1,0])]])
np.savetxt(ROOT/'shield_bh.csv', curve, delimiter=',', header='H_Apm,B_T', comments='')
np.savetxt(ROOT/'shield_curve_pixel_readings.csv', raw, delimiter=',',
           header='x_pixel,y_pixel,H_Apm,B_T', comments='')
provenance = {
    'material': 'VAC MUMETALL, published static magnetization trace',
    'manufacturer': 'VACUUMSCHMELZE',
    'source_url': 'https://vacuumschmelze.com/03_Documents/Brochures/Product%20Information%20Soft%20Magnetic%20Materials%20and%20Semi-Finished%20Products.pdf',
    'edition': '2002 PHT-001', 'pdf_page': 20, 'printed_page': 18, 'figure': 7,
    'measurement_condition': 'Static DC, stamped rings, strip thickness 1 mm, magnetic final annealing',
    'source_pdf_sha256': hashlib.sha256((ROOT/'VAC_soft_curl.pdf').read_bytes()).hexdigest(),
    'image_pixel_size': [7216,6352], 'x_grid_pixels': xgrid.tolist(), 'y_grid_pixels': ygrid.tolist(),
    'x_axis_unit_on_source': 'A/cm', 'H_conversion_to_Apm': 100,
    'trace_selection_anchors_pixels': np.c_[anchor_x,anchor_y].tolist(),
    'retained_chart_knots': len(knots), 'chart_H_range_Apm': knots[[0,-1],0].tolist(),
    'chart_B_range_T': knots[[0,-1],1].tolist(),
    'reading_uncertainty_estimate': 'Approximately 2-3% in B/H around clear trace; larger near overlapping traces. Not a calibrated measurement uncertainty.',
    'assumptions': ['Added origin; linear interpolation in H-B in both solvers',
                    'Constant magnetization above last digitized H, extended to H=200000 A/m',
                    'Isotropic, single-valued, no hysteresis',
                    'Same 1 mm-reference material curve held fixed for 0.10 and 0.20 mm geometric comparison; no claim of actual foil properties',
                    '2002 material trace is not the same data/condition as the 2024 MUMETALL datasheet or Magnetic Shield Corporation MuMETAL foil'],
}
(ROOT/'shield_material_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
plt.rcParams.update({'font.family':'Microsoft YaHei', 'axes.unicode_minus':False})
fig, ax = plt.subplots(figsize=(11,9), constrained_layout=True)
ax.imshow(255-image, cmap='gray', vmin=0, vmax=255)
ax.scatter(raw[:,0], raw[:,1], s=12, facecolor='#d94923', edgecolor='none', label='读取的 MUMETALL 曲线点')
ax.set(xlim=(400,7216), ylim=(5900,700), xlabel='原图 x 像素', ylabel='原图 y 像素',
       title='VAC 静态曲线图 7：MUMETALL 曲线读图核查\nH：A/cm → A/m；用于数值基准，非薄箔实测证书')
ax.legend(loc='lower right');fig.savefig(ROOT/'shield_curve_digitization_check.png', dpi=150)
plt.close(fig)
print(json.dumps(provenance, indent=2), flush=True)
