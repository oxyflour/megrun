"""Read completed AEDT fields, convergence and material audits without saving input."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parent
TOOL=Path(r'C:\Projects\magtense-gpu\repaired-skill\magtense-cupy-magnetostatics\scripts\aedt_magtense_bridge')
sys.path.insert(0,str(TOOL))
from aedt_export import classify_material
from bridge import read_maxwell_fld

def save(path,value):
    path.write_text(json.dumps(value,indent=2,default=str),encoding='utf-8')

parser=argparse.ArgumentParser()
parser.add_argument('--project',type=Path,required=True)
parser.add_argument('--design',required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
start=time.perf_counter();args.out.mkdir(exist_ok=True)
before=hashlib.sha256(args.project.read_bytes()).hexdigest()
points=np.loadtxt(ROOT/'points.csv',delimiter=',',skiprows=1)
with (args.out/'points_mm.pts').open('w') as stream:
    stream.write('Unit=mm\n');np.savetxt(stream,points*1000,fmt='%.15g')
os.environ['ANSYSEM_ROOT222']=r'C:\Program Files\AnsysEM\v222\Win64'
from pyaedt import Maxwell3d
from pyaedt.generic.settings import settings
settings.use_grpc_api=True;settings.enable_error_handler=False
app=Maxwell3d(projectname=str(args.project),designname=args.design,specified_version='2022.2',
              non_graphical=True,new_desktop_session=True,close_on_exit=False)
try:
    for boundary in app.boundaries:
        if boundary.type=='Current':
            assert float(str(boundary.props['Current']).replace('A','').strip())==0
    for method,name in [('export_convergence','convergence.txt'),('export_profile','profile.txt'),
                        ('export_mesh_stats','mesh_stats.txt')]:
        kwargs={'mesh_path' if method=='export_mesh_stats' else 'file_path':str(args.out/name)}
        getattr(app,method)('Reference',**kwargs)
    assert 'Normal Completion' in (args.out/'profile.txt').read_text(errors='replace')
    rows=[]
    for line in (args.out/'convergence.txt').read_text().splitlines():
        if re.match(r'^\s*\d+\|',line):
            parts=[x.strip() for x in line.split('|')]
            rows.append({'pass':int(parts[0]),'tetrahedra':int(parts[1]),
                         'energy_J':float(parts[2]),'energy_error_percent':float(parts[3]),
                         'delta_energy_percent':None if parts[4]=='N/A' else float(parts[4])})
    assert rows
    converged=len(rows)>=3 and all(r['energy_error_percent']<=.5 and
                   r['delta_energy_percent'] is not None and r['delta_energy_percent']<=.5 for r in rows[-2:])
    assert app.post.export_field_file('B',solution='Reference : LastAdaptive',
         output_dir=str(args.out/'maxwell_B.fld'),sample_points_file=str(args.out/'points_mm.pts'),
         export_with_sample_points=True,export_in_si_system=True,reference_coordinate_system='Global',
         export_field_in_reference=True)
    B,scale=read_maxwell_fld(args.out/'maxwell_B.fld',points)
    assert scale==1 and np.isfinite(B).all()
    np.savetxt(args.out/'maxwell_B.csv',np.c_[points,B],delimiter=',',
               header='x_m,y_m,z_m,Bx_T,By_T,Bz_T',comments='')
    mats={'magrun_steel_bh':np.loadtxt(ROOT.parent/'20261107_gap3mm_vacoflux50/vacoflux50_bh.csv',delimiter=',',skiprows=1)}
    if args.design!='Gap_3mm_CCW':
        mats['shield_mumetall_vac_curve']=np.loadtxt(ROOT/'shield_bh.csv',delimiter=',',skiprows=1)
    material_info={}
    for name,expected in mats.items():
        actual=classify_material(app.materials[name])
        np.testing.assert_allclose(actual['bh_H_Apm_B_T'],expected,rtol=1e-12,atol=1e-12)
        material_info[name]={'matches_input_knots':True,'properties':actual}
    # Export working-point B/H at deterministic centers of each nonlinear body.
    sample_model=ROOT/(args.design if args.design!='Gap_3mm_CCW' else 'Shield_T020')/'model.json'
    model=json.loads(sample_model.read_text())
    for obj in model['objects']:
        if obj['properties']['kind']!='nonlinear' or (args.design=='Gap_3mm_CCW' and obj['name']=='MID_SHIELD'):
            continue
        vertices=np.load(sample_model.parent/obj['mesh_npz'])['tetra_m']
        centers=vertices.mean(axis=1)
        centers=centers[np.linspace(0,len(centers)-1,min(200,len(centers)),dtype=int)]
        point_file=args.out/f"{obj['name']}_centers.pts"
        with point_file.open('w') as stream:
            stream.write('Unit=mm\n');np.savetxt(stream,centers*1000,fmt='%.15g')
        fields={}
        for quantity in ['B','H']:
            path=args.out/f"{obj['name']}_{quantity}.fld"
            assert app.post.export_field_file(quantity,solution='Reference : LastAdaptive',output_dir=str(path),
                     sample_points_file=str(point_file),export_with_sample_points=True,export_in_si_system=True)
            fields[quantity],scale=read_maxwell_fld(path,centers)
            assert scale==1 and np.isfinite(fields[quantity]).all()
        h,b=np.linalg.norm(fields['H'],axis=1),np.linalg.norm(fields['B'],axis=1)
        curve=np.asarray(obj['properties']['bh_H_Apm_B_T'])
        target=np.interp(h,curve[:,0],curve[:,1])
        above=h>curve[-1,0]
        target[above]=curve[-1,1]+4e-7*np.pi*(h[above]-curve[-1,0])
        relative=abs(b-target)/np.maximum(target,1e-12)
        material_info[obj['name']]={'samples':len(h),'H_min_Apm':float(h.min()),'H_max_Apm':float(h.max()),
              'B_min_T':float(b.min()),'B_max_T':float(b.max()),
              'BH_relative_difference_median':float(np.median(relative)),
              'BH_relative_difference_p95':float(np.percentile(relative,95)),
              'BH_relative_difference_max':float(relative.max())}
        if obj['name']=='MID_SHIELD':
            for quantity in ['Mag_H','Mag_B']:
                value=app.post.get_scalar_field_value(quantity,scalar_function='Maximum',
                    solution='Reference : LastAdaptive',object_name='MID_SHIELD',object_type='volume')
                material_info[obj['name']]['volume_max_'+quantity]=value
    save(args.out/'constitutive_audit.json',material_info)
    save(args.out/'export_status.json',{'state':'Exported; native convergence audited','design':args.design,
          'project':str(args.project),'point_count':len(points),'coordinate_scale_to_m':1.,
          'adaptive_converged':converged,'passes':rows[-1]['pass'],
          'energy_error_percent':rows[-1]['energy_error_percent'],'total_tetrahedra':rows[-1]['tetrahedra'],
          'adaptive_history':rows,'seconds':time.perf_counter()-start})
finally:
    app.release_desktop(close_projects=True,close_desktop=True)
after=hashlib.sha256(args.project.read_bytes()).hexdigest()
save(args.out/'input_integrity.json',{'sha256_before':before,'sha256_after':after,'unchanged':before==after})
assert before==after, 'Export unexpectedly changed the input project'
print(f'EXPORTED {args.design}',flush=True)
