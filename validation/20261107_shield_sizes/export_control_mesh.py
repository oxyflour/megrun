"""Export a completed field and a coordinate fingerprint of the whole FEM mesh."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
TOOL=Path(r'C:\Projects\magtense-gpu\repaired-skill\magtense-cupy-magnetostatics\scripts\aedt_magtense_bridge')
sys.path.insert(0,str(TOOL))
from aedtplt import read_aedtplt
from bridge import read_maxwell_fld

parser=argparse.ArgumentParser()
parser.add_argument('--project',type=Path,required=True)
parser.add_argument('--design',required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args();args.out.mkdir(exist_ok=True)
before=hashlib.sha256(args.project.read_bytes()).hexdigest()
os.environ['ANSYSEM_ROOT222']=r'C:\Program Files\AnsysEM\v222\Win64'
from pyaedt import Maxwell3d
from pyaedt.generic.settings import settings
settings.use_grpc_api=True;settings.enable_error_handler=False
app=Maxwell3d(projectname=str(args.project),designname=args.design,specified_version='2022.2',
              non_graphical=True,new_desktop_session=True,close_on_exit=False)
try:
    (args.out/'owned_desktop.json').write_text(json.dumps({'pid':app.desktop_class.aedt_process_id,
                                                        'owned_new_session':True}))
    for method,name in [('export_convergence','convergence.txt'),('export_profile','profile.txt'),('export_mesh_stats','mesh_stats.txt')]:
        kwargs={'mesh_path' if method=='export_mesh_stats' else 'file_path':str(args.out/name)}
        getattr(app,method)('Reference',**kwargs)
    assert 'Normal Completion' in (args.out/'profile.txt').read_text(errors='replace')
    if args.design=='Shield_T010':
        extrema={}
        for quantity in ['Mag_H','Mag_B']:
            try:
                value=app.post.get_scalar_field_value(quantity,scalar_function='Maximum',
                        solution='Reference : LastAdaptive',object_name='MID_SHIELD',object_type='volume')
                extrema[quantity]=value
            except Exception as error:
                extrema[quantity]={'unavailable':str(error)}
        (args.out/'shield_volume_field_maxima.json').write_text(json.dumps(extrema,indent=2),encoding='utf-8')
    points=np.loadtxt(ROOT/'points.csv',delimiter=',',skiprows=1)
    with (args.out/'points.pts').open('w') as stream:
        stream.write('Unit=mm\n');np.savetxt(stream,points*1000,fmt='%.15g')
    assert app.post.export_field_file('B',solution='Reference : LastAdaptive',
          output_dir=str(args.out/'maxwell_B.fld'),sample_points_file=str(args.out/'points.pts'),
          export_with_sample_points=True,export_in_si_system=True,reference_coordinate_system='Global',
          export_field_in_reference=True)
    field,scale=read_maxwell_fld(args.out/'maxwell_B.fld',points)
    assert scale==1 and np.isfinite(field).all()
    np.savetxt(args.out/'maxwell_B.csv',np.c_[points,field],delimiter=',',
               header='x_m,y_m,z_m,Bx_T,By_T,Bz_T',comments='')
    plot=app.post.create_fieldplot_volume(app.modeler.solid_names,'Mesh',
           setup='Reference : LastAdaptive',plot_name='GRID_CORNER_FINGERPRINT')
    assert plot
    try:
        raw=app.post.export_field_plot(plot.name,str(args.out),file_name='all_mesh',file_format='aedtplt')
    finally:
        plot.delete()
    vertices,stats=read_aedtplt(raw,1e-3)
    assert len(vertices)>100000
    # Each tetrahedron and the collection are unordered for this geometric check.
    coords=np.rint(vertices/1e-12).astype(np.int64)
    order=np.lexsort((coords[:,:,2],coords[:,:,1],coords[:,:,0]),axis=1)
    ordered=np.take_along_axis(coords,order[:,:,None],axis=1).reshape(len(coords),12)
    packed=np.ascontiguousarray(ordered).view(np.dtype('S96')).ravel()
    packed.sort()
    fingerprint=hashlib.sha256(packed.tobytes()).hexdigest()
    row=[]
    for line in (args.out/'convergence.txt').read_text().splitlines():
        if re.match(r'^\s*\d+\|',line):row=[s.strip() for s in line.split('|')]
    mesh={'tetrahedra':len(vertices),'corner_coordinate_set_sha256':fingerprint,
          'coordinate_rounding_m':1e-12,'quadratic_midside_nodes_not_compared':True,
          'raw_mesh_stats':stats,'native_final_tetrahedra':int(row[1]),
          'passes':int(row[0]),'energy_error_percent':float(row[3]),
          'MID_SHIELD_material':app.modeler['MID_SHIELD'].material_name,
          'Region_bbox_mm':list(app.modeler['Region'].bounding_box)}
    assert len(vertices)==int(row[1])
    (args.out/'mesh_fingerprint.json').write_text(json.dumps(mesh,indent=2),encoding='utf-8')
finally:
    app.release_desktop(close_projects=True,close_desktop=True)
after=hashlib.sha256(args.project.read_bytes()).hexdigest()
assert before==after,'Source project changed during export'
print('EXPORTED AND FINGERPRINTED',args.design,flush=True)
