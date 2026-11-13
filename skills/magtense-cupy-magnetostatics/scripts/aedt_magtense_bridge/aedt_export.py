"""AEDT project -> per-object TET4/TET10 volume mesh + normalized magnetic materials.

Must run on Windows with licensed AEDT / a solved 3D Magnetostatic design.
"""
from __future__ import annotations

import json
import inspect
from pathlib import Path
import re
import warnings
import hashlib
from types import SimpleNamespace

import numpy as np

from aedtplt import read_aedtplt, get_unit_scale

MU0 = 4e-7 * np.pi


class ConversionError(RuntimeError):
    pass


def _quantity(x, quantity='') -> float:
    """Read simple AEDT numeric+unit; refuse evaluated expressions rather than guess."""
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    m = re.fullmatch(r'([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z_0-9/]*)', s)
    if not m:
        raise ConversionError(f'Non-numeric AEDT expression {x!r}. Provide an override.')
    v, u = float(m.group(1)), m.group(2).lower()
    if quantity == 'H':
        if u in ('', 'a_per_meter', 'a/m', 'apermeter', 'a_per_m'):
            return v
        if u in ('ka_per_meter', 'ka/m', 'ka_per_m'):
            return 1000*v
    elif quantity == 'B':
        if u in ('', 't', 'tesla'):
            return v
        if u in ('mt', 'millitesla'):
            return v * 1e-3
        if u in ('gauss', 'g'):
            return v * 1e-4
    elif not u:
        return v
    raise ConversionError(f'Unsupported unit {u!r} for {quantity} in {x!r}')


def _convert_bh(value, hunit, bunit):
    rows = []
    for pair in value:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ConversionError('B-H must be [H,B] numeric pairs')
        h = _quantity(pair[0], 'H')
        b = _quantity(pair[1], 'B')
        h_unit = str(hunit or 'A_per_meter').lower()
        b_unit = str(bunit or 'tesla').lower()
        if h_unit in ('ka_per_meter', 'ka/m'):
            h *= 1000.0
        elif h_unit not in ('a_per_meter','a/m',''):
            raise ConversionError(f'Unsupported H-unit: {hunit!r}')
        if b_unit in ('mt', 'millitesla'):
            b *= 1e-3
        elif b_unit not in ('tesla','t',''):
            raise ConversionError(f'Unsupported B-unit: {bunit!r}')
        rows.append([h, b])
    rows.sort(key=lambda p: p[0])
    if not rows or rows[0][0] != 0 or abs(rows[0][1]) > 1e-12:
        raise ConversionError('B-H is not a virgin single-valued curve starting at [0,0]. Override needed.')
    return rows


def _coercivity(material):
    """Read signed Hc vector from original AEDT material props, preserving its sign."""
    props = getattr(material, '_props', {}) or {}
    raw = props.get('magnetic_coercivity', None)
    if raw is None:
        return np.zeros(3)
    if not isinstance(raw, dict):
        raise ConversionError('Unexpected magnetic_coercivity representation: add override')
    magnitude = _quantity(raw.get('Magnitude', 0), 'H')
    if abs(magnitude) < 1e-8:
        return np.zeros(3)
    xyz = np.array([_quantity(raw.get(f'DirComp{i}', 0)) for i in range(1, 4)])
    n = np.linalg.norm(xyz)
    if n == 0:
        raise ConversionError('Nonzero Hc with zero direction')
    return magnitude * xyz / n


def classify_material(material):
    """Return normalized isotropic material properties or kind='skip'."""
    name = str(getattr(material, 'name', '')).lower()
    if name.strip().strip('"') in ('air', 'vacuum'):
        return {'kind': 'skip', 'reason': 'air/vacuum'}
    p = getattr(material, 'permeability', None)
    if p is None:
        raise ConversionError(f'No permeability property on material {name}')
    tp = getattr(p, 'type', 'simple')
    hc = _coercivity(material)
    if tp == 'nonlinear':
        if np.linalg.norm(hc) > 1e-8:
            raise ConversionError(f'Nonlinear B-H plus coercivity in {name} not supported')
        if getattr(p, 'is_temperature_dependent', False):
            raise ConversionError(f'Temperature-dependent BH in {name} not supported')
        data = _convert_bh(p.value, getattr(p, 'hunit', 'A_per_meter'),
                           getattr(p, 'bunit', 'tesla'))
        return {'kind':'nonlinear', 'bh_H_Apm_B_T':data}
    if tp != 'simple':
        raise ConversionError(f'{name}: {tp} permeability not isotropic; use override or extend adapter')
    mur = _quantity(p.value)
    if mur < 1:
        raise ConversionError(f'{name}: diamagnet mu_r={mur} not supported')
    if np.linalg.norm(hc) > 1e-8:
        # Maxwell coercivity H_c (negative along remanent easy-axis for typical PM);
        # linear-recoil: B_r = -mu0 mu_r H_c vector.
        mrvec = -mur*hc
        return {'kind':'permanent', 'br_T':float(MU0*np.linalg.norm(mrvec)),
                'direction':(mrvec/np.linalg.norm(mrvec)).tolist(), 'mu_r':mur}
    if abs(mur - 1) < 1e-7:
        return {'kind':'skip', 'reason':'nonmagnetic mu_r≈1, Hc=0'}
    return {'kind':'linear', 'mu_r':mur}


def _resolve_properties(name, material_name, material, overrides):
    obj_over = overrides.get('objects', {}).get(name)
    mat_over = overrides.get('materials', {}).get(material_name)
    if obj_over is not None:
        return obj_over, 'object_override'
    if mat_over is not None:
        return mat_over, 'material_override'
    return classify_material(material), 'aedt'


def _line_points_mm(spec: str):
    try:
        a, b, number = spec.split(':')
        start = np.array([float(v) for v in a.split(',')])
        stop = np.array([float(v) for v in b.split(',')])
        count = int(number)
    except (ValueError, AttributeError) as e:
        raise ConversionError('Use --line-mm "x0,y0,z0:x1,y1,z1:count"') from e
    if start.shape != (3,) or stop.shape != (3,) or count < 2:
        raise ConversionError('Bad --line-mm: need 3+3 coordinates and count>=2')
    return np.linspace(start, stop, count)*1e-3


def load_points(points_csv: str | None, line_mm: str | None):
    if bool(points_csv) == bool(line_mm):
        raise ConversionError('Choose exactly one of --points-csv and --line-mm')
    if line_mm:
        return _line_points_mm(line_mm)
    # CSV coordinates always in meters; allow optional header x_m,y_m,z_m.
    try:
        p = np.loadtxt(points_csv, delimiter=',', ndmin=2)
    except ValueError:
        p = np.loadtxt(points_csv, delimiter=',', skiprows=1, ndmin=2)
    if p.shape[1] != 3 or not np.all(np.isfinite(p)):
        raise ConversionError('points CSV must contain x_m,y_m,z_m')
    return p


def _check_current_sources(app, skip_check=False):
    sources = []
    for bound in getattr(app, 'boundaries', []):
        t = str(getattr(bound, 'type', '')).lower()
        if any(q in t for q in ('current', 'winding', 'coil', 'externalcircuit')):
            sources.append({'name': str(getattr(bound, 'name', '')), 'type':t})
    if sources and not skip_check:
        raise ConversionError(f'Current/winding excitations are NOT modeled in this adapter: {sources}. '
                              'Use --ignore-current-check only if you verified all currents=0.')
    return sources


def _validate_mesh_units_against_cad(obj, vertices_m, stats, model_unit_scale):
    """Match mesh coordinates with AEDT object's bounding box (usually model units).

    AEDT versions may export mesh Nodes() in SI or model units. Detect the
    1e3 ambiguity using the *actual* CAD object's bounding box. If no usable
    bounding box is available, keep model-unit scaling and record the warning.
    """
    try:
        bb = np.array([float(x) for x in obj.bounding_box],dtype=float)
        if bb.shape!=(6,) or not np.all(np.isfinite(bb)):
            raise ValueError('invalid bounding box')
        cad_lo, cad_hi = bb[:3]*model_unit_scale,bb[3:]*model_unit_scale
        cad_size = np.linalg.norm(cad_hi-cad_lo)
        if cad_size <= 0:
            raise ValueError('empty CAD object bbox')
        options = [(1.0,'model_units')]
        if model_unit_scale != 1.0:
            options.append((1/model_unit_scale,'SI_meters'))
        scores = []
        for factor,name in options:
            mlo=vertices_m.min(axis=(0,1))*factor
            mhi=vertices_m.max(axis=(0,1))*factor
            # Normalize bbox discrepancy by CAD overall size.
            score=(np.linalg.norm(mlo-cad_lo)+np.linalg.norm(mhi-cad_hi))/cad_size
            scores.append((score,factor,name))
        score,factor,name=min(scores)
        if score>0.8:
            raise ConversionError(f'{obj.name}: exported volume mesh bounding box does not match CAD '
                                  f'under model- or SI-unit scaling. scores={scores}. '
                                  'Check mesh units and AEDT solution/variation.')
        if factor!=1.0:
            warnings.warn(f'{obj.name}: AEDTPLT nodes appear to be in SI meters, not model units; '
                          'rescaling automatically using CAD bounding box.')
            vertices_m *= factor
            stats['volume_m3'] *= factor**3
            stats['bounds_min_m']=(np.array(stats['bounds_min_m'])*factor).tolist()
            stats['bounds_max_m']=(np.array(stats['bounds_max_m'])*factor).tolist()
        stats['node_units_assumption']=name
        stats['cad_bbox_relative_mismatch']=float(score)
    except (AttributeError, TypeError, ValueError) as ex:
        warnings.warn(f'{getattr(obj,"name","object")}: cannot validate AEDTPLT length units '
                      f'against AEDT CAD bbox ({ex}). Assuming model units; check coordinates.')
        stats['node_units_assumption']='unverified_model_units'
    return vertices_m,stats


def _native_mesh_units(obj,vertices,stats,unit_scale):
    """CAD volume also supports spheres/cylinders with incomplete CAD vertices."""
    volume=obj.get('cad_volume_model_units3')
    if volume is not None and np.isfinite(volume) and volume>0:
        target=volume*unit_scale**3
        factors=[1.] if unit_scale==1. else [1.,1/unit_scale]
        score,factor=min((abs(stats['volume_m3']*f**3/target-1),f) for f in factors)
        if score>.8:
            raise ConversionError(obj['name']+': native mesh volume does not match CAD in model or SI units')
        vertices*=factor;stats['volume_m3']*=factor**3
        stats['bounds_min_m']=(np.asarray(stats['bounds_min_m'])*factor).tolist()
        stats['bounds_max_m']=(np.asarray(stats['bounds_max_m'])*factor).tolist()
        stats.update(cad_volume_relative_mismatch=float(score),cad_units_verification='native CAD volume',
            node_units_assumption='SI_meters' if factor!=1 else 'model_units')
        return vertices,stats
    points=np.asarray(obj['cad_vertex_positions'],dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise ConversionError(obj['name']+': cannot verify mesh units; no CAD volume or usable vertices')
    proxy=SimpleNamespace(name=obj['name'],bounding_box=np.r_[points.min(axis=0),points.max(axis=0)])
    vertices,stats=_validate_mesh_units_against_cad(proxy,vertices,stats,unit_scale)
    if stats['node_units_assumption']=='unverified_model_units':
        raise ConversionError(obj['name']+': native mesh units are not verified')
    stats['cad_units_verification']='native CAD vertices; curved extrema may lie between vertices'
    return vertices,stats


def export_project(project: str, design: str, out: str, points_csv=None, line_mm=None,
                   setup=None, version=None, objects=None, overrides_path=None,
                   max_tiles=10000, no_reference=False, ignore_current_check=False,
                   new_desktop=False):
    legacy = False
    try:
        from ansys.aedt.core import Maxwell3d
    except ImportError:
        try:
            from pyaedt import Maxwell3d
            from pyaedt.generic.settings import settings
            settings.use_grpc_api = True
            legacy = True
        except ImportError as e:
            raise ConversionError('Install PyAEDT on an AEDT-equipped Windows computer') from e
    root = Path(out).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / 'mesh').mkdir(exist_ok=True)
    overrides = json.loads(Path(overrides_path).read_text(encoding='utf-8')) if overrides_path else {}
    points = load_points(points_csv, line_mm)
    np.savetxt(root/'points.csv', points, delimiter=',', header='x_m,y_m,z_m',comments='')
    params = dict(project=str(Path(project).resolve()), design=design,
                  non_graphical=True, new_desktop=new_desktop)
    if version:
        params['version'] = version
    if legacy:
        params = dict(projectname=params['project'],designname=design,non_graphical=True,
                      new_desktop_session=new_desktop,
                      **({'specified_version':version} if version else {}))
    app = Maxwell3d(**params)
    try:
        solution_type = str(getattr(app, 'solution_type', ''))
        if 'magnetostatic' not in solution_type.lower():
            raise ConversionError(f'Design {design}: solution_type={solution_type!r}, requires Magnetostatic')
        setup = setup or str(app.nominal_adaptive)
        unit = str(app.modeler.model_units)
        unit_scale = get_unit_scale(unit)
        # AEDT 2022 R2 can ignore a Unit=m header and read numbers in model units.
        # Matching both the header and numbers to the actual model is safe in either interpretation.
        with (root/'points.pts').open('w',encoding='utf-8') as f:
            f.write(f'Unit={unit}\n')
            for p in points/unit_scale:
                f.write(f'{p[0]:.15g} {p[1]:.15g} {p[2]:.15g}\n')
        sources = _check_current_sources(app, ignore_current_check)
        selected = list(objects) if objects else list(app.modeler.object_names)
        model = {'schema_version': 1,
                 'source': {'project':str(project), 'design':design, 'solution':setup,
                            'model_unit':unit, 'source_check':sources},
                 'objects': [], 'skipped': [], 'points':'points.csv',
                 'maxwell_field':'maxwell_B.fld', 'mesh_unit':'m'}
        total = 0
        for idx, name in enumerate(selected):
            try:
                obj = app.modeler[name]
            except Exception as e:
                raise ConversionError(f'Object not found in model: {name}') from e
            is_solid = getattr(obj,'is_3d',None)
            if is_solid is None:
                is_solid = str(getattr(obj,'object_type','')).lower() == 'solid'
            if not is_solid:
                model['skipped'].append({'name':name,'reason':'not 3D solid'})
                continue
            material_name = str(obj.material_name).strip('"')
            if material_name.lower() in ('air', 'vacuum'):
                model['skipped'].append({'name':name,'material':material_name,'reason':'air/vacuum'})
                continue
            try:
                material = app.materials[material_name]
            except Exception as e:
                if name not in overrides.get('objects', {}) and material_name not in overrides.get('materials', {}):
                    raise ConversionError(f'Material not found: {material_name} for {name}') from e
                material = None
            if material is None and name not in overrides.get('objects', {}) and material_name not in overrides.get('materials', {}):
                raise ConversionError(f'No material definition for {material_name}')
            props, origin = _resolve_properties(name, material_name, material, overrides)
            if props['kind'] == 'skip':
                model['skipped'].append({'name':name,'material':material_name,
                                         'reason':props.get('reason','nonmagnetic')})
                continue
            if props['kind'] not in ('permanent','nonlinear','linear'):
                raise ConversionError(f'Unsupported kind={props["kind"]!r} for {name}')
            cs = getattr(obj, 'part_coordinate_system', 'Global')
            if props['kind'] == 'permanent' and cs not in (None,'Global') and origin == 'aedt':
                raise ConversionError(f'{name}: magnet coordinate system={cs!r} not Global. '
                                      'Give global remanence direction using --overrides.')
            if not obj.solve_inside:
                raise ConversionError(f'{name}: Solve Inside is disabled, cannot extract its 3D volume mesh')
            safe = re.sub(r'[^A-Za-z0-9_-]+', '_',name)[:55]
            plot_name = f'MT_TEMP_{idx:04d}'
            plot = app.post.create_fieldplot_volume(assignment=[name],quantity='Mesh',
                                                   setup=setup,plot_name=plot_name)
            if not plot:
                raise ConversionError(f'Cannot create VOLUME mesh plot for {name}')
            try:
                raw_path = app.post.export_field_plot(plot_name=plot.name,
                    output_dir=root/'mesh',file_name=f'{idx:04d}_{safe}',file_format='aedtplt')
            finally:
                plot.delete()
            if not raw_path or not Path(raw_path).exists():
                raise ConversionError(f'Cannot export .aedtplt file for {name}')
            v, stats = read_aedtplt(raw_path,unit_scale)
            v, stats = _validate_mesh_units_against_cad(obj,v,stats,unit_scale)
            total += len(v)
            print(f'  {name} ({material_name}, {props["kind"]}): {len(v):,} tetra, {stats["volume_m3"]:.6g} m^3')
            if stats['curved']:
                warnings.warn(f'{name}: curved tetrahedral geometry linearized to 4 corner nodes.')
            if total > max_tiles:
                raise ConversionError(f'Exceeded --max-tiles {max_tiles:,}: {total:,}. '
                                      'Use a coarser Maxwell volume mesh or increase the limit.')
            mesh_npz = root/'mesh'/f'{idx:04d}_{safe}.npz'
            np.savez_compressed(mesh_npz, tetra_m=v)
            model['objects'].append({'name':name,'material':material_name,
                   'properties':props,'properties_source':origin,
                   'mesh_npz':str(mesh_npz.relative_to(root)).replace('\\','/'),
                   'mesh_raw':str(Path(raw_path).relative_to(root)).replace('\\','/'),
                   'mesh_stats':stats})
        if not total:
            raise ConversionError('No magnetic solids were exported.')
        model['total_magnetic_tetra'] = total
        (root/'model.json').write_text(json.dumps(model,indent=2,ensure_ascii=False),encoding='utf-8')
        print(f'Saved {len(model["objects"])} magnetic solids / {total:,} tetra, skipped {len(model["skipped"])}.')
        if not no_reference:
            ref = root/'maxwell_B.fld'
            output_keyword = 'output_file' if 'output_file' in inspect.signature(app.post.export_field_file).parameters else 'output_dir'
            ok = app.post.export_field_file(quantity='B',solution=setup,
                        **{output_keyword:str(ref)},sample_points_file=str(root/'points.pts'),
                        export_with_sample_points=True, reference_coordinate_system='Global',
                        export_in_si_system=True, export_field_in_reference=True)
            if not ok or not ref.exists():
                raise ConversionError('Maxwell B_Vector field export failed. '
                        'Check solved setup, field name, and sample points. '
                        'You can retry with --no-reference and export .fld manually.')
            from bridge import read_maxwell_fld
            _, coordinate_scale = read_maxwell_fld(ref,points)
            if coordinate_scale != 1.0:
                raise ConversionError('Maxwell output XYZ are not SI-aligned with the requested points')
            print('Saved Maxwell reference:',ref)
        return model
    finally:
        # Only a session explicitly created for this export belongs to this call.
        if new_desktop:
            app.release_desktop(close_projects=True,close_desktop=True)


def _native_arguments(values):
    """Decode named AEDT COM arrays, retaining repeated children (BH points)."""
    values=list(values)
    if not values or not str(values[0]).startswith('NAME:'):
        raise ConversionError('Malformed native material argument array')
    body=values[1:]
    if not any(isinstance(x,list) or isinstance(x,str) and x.endswith(':=') for x in body):
        return body
    result={};repeated=set();i=0
    def add(key,value):
        if key in result:
            if key not in repeated:
                result[key]=[result[key]];repeated.add(key)
            result[key].append(value)
        else:
            result[key]=value
    while i<len(body):
        item=body[i]
        if isinstance(item,list) and item and str(item[0]).startswith('NAME:'):
            add(str(item[0])[5:],_native_arguments(item));i+=1
        elif isinstance(item,str) and item.endswith(':=') and i+1<len(body):
            add(item[:-2],body[i+1]);i+=2
        else:
            raise ConversionError(f'Malformed native material item {item!r}')
    return result


def _native_material(name,raw):
    properties=_native_arguments(raw)
    if not isinstance(properties,dict) or 'permeability' not in properties:
        raise ConversionError(f'Native material {name} has no explicit permeability')
    if properties.get('CoordinateSystemType','Cartesian')!='Cartesian':
        raise ConversionError(f'{name}: non-Cartesian material axes require a global normalized override')
    if properties.get('ModifierData'):
        raise ConversionError(f'{name}: material modifiers require a normalized override')
    permeability=properties['permeability'];kind='simple';value=permeability;hunit='A_per_meter';bunit='tesla';temperature=False
    if isinstance(permeability,dict):
        kind=str(permeability.get('property_type','unsupported'))
        if kind=='nonlinear':
            coordinates=permeability.get('BHCoordinates',{})
            if 'Point' in coordinates:
                value=coordinates['Point']
                if value and not isinstance(value[0],list):value=[value]
            elif 'Points' in coordinates:
                flat=coordinates['Points']
                if len(flat)%2:raise ConversionError('Odd number of native BH coordinates')
                value=[flat[i:i+2] for i in range(0,len(flat),2)]
            else:
                raise ConversionError(f'{name}: no single-valued native BH points')
            hunit=permeability.get('HUnit','A_per_meter');bunit=permeability.get('BUnit','tesla')
            temperature=permeability.get('IsTemperatureDependent',False)
            if str(temperature).lower() not in ('false','0'):temperature=True
            else:temperature=False
            if permeability.get('BTypeForSingleCurve','normal').lower()!='normal':
                raise ConversionError(f'{name}: non-normal BH curve requires an override')
    return SimpleNamespace(name=name,_props=properties,permeability=SimpleNamespace(type=kind,value=value,
        hunit=hunit,bunit=bunit,is_temperature_dependent=temperature))


def _native_scalar(value,variables):
    """Resolve exact variable aliases only; never eval project expressions."""
    seen=set()
    while isinstance(value,str) and value in variables:
        if value in seen:raise ConversionError('Cyclic native variable alias: '+value)
        seen.add(value);value=variables[value]
    return value


def _native_physics(info):
    """Retain all native physics, reject features the offline solver omits."""
    variables=info.get('variables',{})
    for item in info.get('excitations',[]):
        properties=item.get('properties',{});kind=item['type'].lower()
        if kind not in ('current','winding') or 'Current' not in properties:
            raise ConversionError(f'Unsupported native excitation {item["name"]}: {item["type"]}; archived but not modeled')
        if kind=='winding' and properties.get('Type','Current').lower()!='current':
            raise ConversionError('Voltage/external-circuit windings are not modeled')
        value=str(_native_scalar(properties['Current'],variables)).strip()
        match=re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(A|mA|kA|amp|ampere)?',value)
        if not match or not np.isfinite(float(match[1])) or float(match[1])!=0:
            raise ConversionError(f'{item["name"]}: current {value!r} is nonzero or unresolved; offline current sources are not implemented')
    for item in info.get('boundaries',[]):
        kind=str(item['properties'].get('Type','')).lower()
        if kind!='balloon':
            raise ConversionError(f'Native boundary {item["name"]} ({kind or "unknown type"}) cannot be represented by infinite-air FEM-BEM')


def import_native_bundle(bundle,out,*,points_csv=None,line_mm=None,overrides_path=None,max_tiles=10000):
    """Verify a native disk archive and create self-contained existing inputs.

    No AEDT/PyAEDT imports or license are used. Raw all-domain mesh and settings
    stay in the transfer bundle; model.json embeds native metadata, while only
    supported magnetic cells become canonical SI mesh NPZs for offline solving.
    """
    from aedt_export_native import _bool_value
    root=Path(bundle).resolve();target=Path(out).resolve()
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8-sig'))
    if manifest.get('schema')!='aedt-native-disk-1' or manifest.get('state')!='complete':
        raise ConversionError('Native archive is incomplete or has an unknown schema')
    files=manifest.get('files',{})
    def member(relative):
        if not isinstance(relative,str) or '\\' in relative or ':' in relative:
            raise ConversionError('Invalid native archive member path')
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or path==root or '..' in Path(relative).parts:
            raise ConversionError('Native archive path escapes its root: '+relative)
        if relative not in files:raise ConversionError('Unhashed native archive member: '+relative)
        return path
    for relative,spec in files.items():
        path=member(relative)
        if not path.is_file() or path.stat().st_size!=spec['bytes']:
            raise ConversionError('Native archive file missing/size mismatch: '+relative)
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=spec['sha256']:raise ConversionError('Native archive SHA-256 mismatch: '+relative)
    info=json.loads(member('native_export.json').read_text(encoding='utf-8-sig'))
    if info.get('state')!='complete' or not info.get('source_project_unchanged'):
        raise ConversionError('Native archive has no completed source integrity check')
    member('native/source_project.aedt')
    if files['native/source_project.aedt']['sha256']!=info['source']['project_sha256']:
        raise ConversionError('Saved native project copy does not match the exported source hash')
    _native_physics(info)
    if points_csv or line_mm:
        points=load_points(points_csv,line_mm)
    elif info.get('points'):
        points=load_points(str(member(info['points'])),None)
    else:
        raise ConversionError('Provide --points-csv or --line-mm; native bundle contains no observation points')
    overrides=json.loads(Path(overrides_path).read_text(encoding='utf-8')) if overrides_path else {}
    # Validate all material/current/coordinate-system constraints before output.
    selection=[];skipped=[];unit_scale=get_unit_scale(info['model_unit'])
    for obj in info['objects']:
        name=obj['name'];material_name=obj['material'];attributes=obj['attributes']
        if obj['group']!='Solids' or not _bool_value(attributes.get('Model',True)):
            skipped.append(dict(name=name,reason='not a model solid'));continue
        if material_name.lower() in ('air','vacuum'):
            skipped.append(dict(name=name,material=material_name,reason='air/vacuum'));continue
        overridden=name in overrides.get('objects',{}) or material_name in overrides.get('materials',{})
        material=None if overridden else _native_material(material_name,info['materials'][material_name])
        props,origin=_resolve_properties(name,material_name,material,overrides)
        if props['kind']=='skip':
            skipped.append(dict(name=name,material=material_name,reason=props.get('reason','nonmagnetic')));continue
        if props['kind'] not in ('permanent','nonlinear','linear'):
            raise ConversionError('Unsupported normalized native material: '+name)
        if props['kind']=='permanent' and origin=='aedt' and attributes.get('Orientation','Global')!='Global':
            raise ConversionError(name+': local magnet orientation requires an explicit global-direction override')
        if not _bool_value(attributes.get('Solve Inside',False)) or not obj.get('mesh_raw'):
            raise ConversionError(name+': no solved interior volume mesh')
        member(obj['mesh_raw']);selection.append((obj,props,origin))
    if not selection:raise ConversionError('No supported magnetic volume cells in native bundle')
    if target.exists() and any(target.iterdir()):
        raise ConversionError('Import output must be absent or empty; existing data is retained')
    target.mkdir(parents=True,exist_ok=True);(target/'mesh').mkdir()
    model=dict(schema_version=1,source=dict(info['source'],native_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        source_check=info['excitations'],native_saved_context=info['saved_context'],native_variables=info['variables'],
        native_boundaries=info['boundaries'],native_export_scope=info['capture_scope'],
        offline_exterior='Infinite air; native finite air region/settings retained as context'),
        objects=[],skipped=skipped,points='points.csv',mesh_unit='m')
    total=0
    for index,(obj,props,origin) in enumerate(selection):
        tetra,stats=read_aedtplt(member(obj['mesh_raw']),unit_scale)
        tetra,stats=_native_mesh_units(obj,tetra,stats,unit_scale)
        total+=len(tetra)
        if total>max_tiles:raise ConversionError(f'Exceeded --max-tiles {max_tiles}: {total}')
        relative=f'mesh/{index:05d}.npz';np.savez_compressed(target/relative,tetra_m=tetra)
        model['objects'].append(dict(name=obj['name'],material=obj['material'],properties=props,
            properties_source='native_'+origin,mesh_npz=relative,mesh_stats=stats,
            mesh_npz_sha256=hashlib.sha256((target/relative).read_bytes()).hexdigest(),
            source_mesh_sha256=files[obj['mesh_raw']]['sha256']))
    np.savetxt(target/'points.csv',points,delimiter=',',header='x_m,y_m,z_m',comments='')
    model['points_sha256']=hashlib.sha256((target/'points.csv').read_bytes()).hexdigest()
    same_points=bool(info.get('points')) and np.array_equal(points,load_points(str(member(info['points'])),None))
    if 'B' in info.get('reference_fields',{}) and same_points:
        from bridge import read_maxwell_fld
        reference=member(info['reference_fields']['B']);B,scale=read_maxwell_fld(reference,points)
        if scale!=1.:raise ConversionError('Native reference coordinates are not SI aligned')
        (target/'maxwell_B.fld').write_bytes(reference.read_bytes());model['maxwell_field']='maxwell_B.fld'
    model['total_magnetic_tetra']=total
    (target/'model.json').write_text(json.dumps(model,indent=2,ensure_ascii=False),encoding='utf-8')
    return model
