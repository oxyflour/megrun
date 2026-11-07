"""Prepare exact-volume shield meshes and common-air observations, without solving."""
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'20261107_gap3mm_vacoflux50/coarse_export'
TOOL=Path(r'C:\Projects\magtense-gpu\skills\magtense-cupy-magnetostatics\scripts\aedt_magtense_bridge')
sys.path.insert(0,str(TOOL))
from bridge import check_points_in_air
from magtense_adapter import assemble_tiles
from reuse_camera_operator import box_tetra

data=json.loads((ROOT/'prepared_cases.json').read_text())
base=json.loads((BASE/'model.json').read_text())
curve=np.loadtxt(ROOT/'shield_bh.csv',delimiter=',',skiprows=1)
points=np.loadtxt(ROOT/'points.csv',delimiter=',',skiprows=1)
labels=json.loads((ROOT/'point_groups.json').read_text())['labels']
prepared=[]
for case in data['cases']:
    for step in ([.8,.4] if case['name']=='Shield_T020' else [.8]):
        name=case['name']+('_refined' if step==.4 else '')
        out=ROOT/name
        out.mkdir(exist_ok=True)
        mesh=out/'mesh';mesh.mkdir(exist_ok=True)
        vertices,stats=box_tetra(case['origin_mm'],
                 [case['thickness_mm'],case['width_mm'],case['height_mm']],step)
        np.savez_compressed(mesh/'MID_SHIELD.npz',tetra_m=vertices)
        model=json.loads(json.dumps(base))
        for obj in model['objects']:
            obj['mesh_npz']=str(BASE/obj['mesh_npz'])
        model['objects'].append({'name':'MID_SHIELD','material':case['material'],
             'properties':{'kind':'nonlinear','bh_H_Apm_B_T':curve.tolist()},
             'properties_source':'VAC chart digitization, shield_material_provenance.json',
             'mesh_npz':'mesh/MID_SHIELD.npz','mesh_stats':{'tetra_count':len(vertices),'curved':False,**stats}})
        model['total_magnetic_tetra']+=len(vertices)
        model['source']={**base['source'],'description':'Original frozen camera mesh plus exact-volume structured shield TET4 mesh',
                          'native_status':'Shield reference not solved yet', 'current_A_turn':0}
        model['reference_source']={k:case[k] for k in ['project','design','setup']}
        model['shield_geometry']={**case,'mesh':stats}
        (out/'model.json').write_text(json.dumps(model,indent=2),encoding='utf-8')
        shutil.copy2(ROOT/'points.csv',out/'points.csv')
        check_points_in_air(out,model,points)
        tiles,curves,tile_stats=assemble_tiles(model['objects'],out)
        assert len(curves)==2, 'Camera and shield must retain distinct curves'
        prepared.append({'directory':str(out),'case':case['name'],'step_mm':step,
                         'magnetic_tiles':tiles.n,'shield_tiles':len(vertices),
                         'distinct_BH_curves':len(curves),'all_566_points_verified_in_air':True,
                         'full_K_float64_GB':72*tiles.n**2/1e9,'shield_mesh':stats})
(ROOT/'export_models.json').write_text(json.dumps(prepared,indent=2),encoding='utf-8')
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
ax=axes[0]
for x,label in [(0,'Camera A'),(11.8,'Camera B')]:
    ax.add_patch(Rectangle((x-4.4,-4.4),8.8,8.8,fc='#d9e2ed',ec='#294563'))
    ax.text(x,0,label,ha='center',va='center')
ax.add_patch(Rectangle((5.8,-5),.2,10,fc='#cf5f2a',ec='#cf5f2a'))
ax.axvline(5.2,color='#267a75',ls=':',label='两侧测线 x=5.2 / 6.6 mm')
ax.axvline(6.6,color='#267a75',ls=':')
ax.set(xlim=(-6,18),ylim=(-6.8,6.2),xlabel='x (mm)',ylabel='y (mm)',title='俯视：3 mm 间隙中央竖直隔磁片')
ax.set_aspect('equal');ax.legend(fontsize=9,loc='lower center')
ax=axes[1]
ax.add_patch(Rectangle((-5,.6),10,5,fc='#fbdfcc',ec='#cf5f2a',label='0.20×10×5 mm'))
ax.add_patch(Rectangle((-4.4,1.4),8.8,3.4,fc='#b7d7d4',ec='#267a75',label='0.10 / 0.20×8.8×3.4 mm'))
ax.set(xlim=(-6,6),ylim=(-.2,6.5),xlabel='y (mm)',ylabel='z (mm)',title='隔磁片正视：厚度沿 x，中心 z=3.1 mm')
ax.set_aspect('equal');ax.legend(fontsize=9,loc='lower center')
fig.savefig(ROOT/'shield_geometry_plan.png',dpi=160);plt.close(fig)
print(json.dumps(prepared,indent=2),flush=True)
