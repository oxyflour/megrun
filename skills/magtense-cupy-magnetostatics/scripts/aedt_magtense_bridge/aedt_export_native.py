# -*- coding: utf-8 -*-
"""Native AEDT 18.x/IronPython 2.7 -> transferable disk bundle, no PyAEDT.

Run with Tools > Run Script; Script Arguments is the absolute path to a JSON
config containing project, design, solution ("Setup : LastAdaptive"), out and
optionally points_csv (SI metres). The design needs an existing generated
volume mesh; explicitly requested B/H references also require saved fields.
This entry only reads the saved project
and creates/deletes temporary Mesh field plots; it never solves or saves it.
All model/Solve-Inside solids, including air, are archived, together with
material definitions, variables, assignments and original solver settings.
bridge.py import-native verifies hashes and creates the existing solver input.
This archives inputs/native evidence, not Maxwell's internal factorization.

Keep this file Python 2.7 compatible and standard-library-only: AEDT 18's
embedded interpreter cannot import the Python 3/NumPy/PyAEDT export module.
"""
from __future__ import print_function

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
import uuid

try:
    string_types = (basestring,)
    integer_types = (int, long)
except NameError:
    string_types = (str,)
    integer_types = (int,)


def _text(value):
    # IronPython aliases unicode/str as types but treats direct unicode(...)
    # specially. An aliased constructor can unexpectedly use ASCII encoding.
    return u'{0}'.format(value)


def _plain(value):
    """Serialize COM/.NET arrays without opaque repr strings."""
    if value is None:
        return None
    if isinstance(value, bool):
        return True if value else False
    if isinstance(value, string_types):
        return u'{0}'.format(value)
    if isinstance(value, integer_types):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, dict):
        return dict((_text(k), _plain(v)) for k, v in value.items())
    try:
        return [_plain(v) for v in value]
    except TypeError:
        return _text(value)


def _write(path, data):
    encoded = json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False)
    json.loads(encoded)  # IronPython COM scalars must produce strict JSON.
    with io.open(path, 'w', encoding='utf-8') as stream:
        stream.write(encoded)


def _bool_value(value):
    if _text(value).lower() in ('true', '1'):
        return True
    if _text(value).lower() in ('false', '0'):
        return False
    raise ValueError('Unrecognized native boolean: ' + _text(value))


def _sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        while True:
            data = stream.read(8 * 1024 * 1024)
            if not data:
                break
            digest.update(data)
    return digest.hexdigest()


def _properties(obj, tab, server):
    result = {}
    for name in obj.GetProperties(tab, server):
        result[_text(name)] = _plain(obj.GetPropertyValue(tab, server, name))
    return result


def _blocks(text, name):
    """Exact balanced saved-project blocks; preserve expressions and all keys."""
    result = []; start = None; depth = 0; offset = 0
    for line in text.splitlines(True):
        opening = re.match(r"\s*\$begin\s+'([^']+)'", line)
        closing = re.match(r"\s*\$end\s+'([^']+)'", line)
        if start is None and opening and opening.group(1) == name:
            start = offset; depth = 0
        if start is not None:
            if opening:
                depth += 1
            if closing:
                depth -= 1
                if depth == 0:
                    result.append(text[start:offset + len(line)]); start = None
        offset += len(line)
    if start is not None:
        raise ValueError('Unbalanced saved project block: ' + name)
    return result


def _saved_context(project_path, design):
    # AEDT files contain binary geometry sections. IronPython's io decoder
    # rejects some of those bytes even with errors='replace'; bytes.decode
    # preserves the surrounding native text blocks on both Python runtimes.
    with open(project_path, 'rb') as stream:
        text = stream.read().decode('utf-8', 'replace')
    designs = [s for s in _blocks(text, 'Maxwell3DModel')
               if re.search(r"(?m)^\s*Name='" + re.escape(design) + r"'\s*$", s[:600])]
    if len(designs) != 1:
        raise ValueError('Cannot locate exactly one saved Maxwell3DModel for ' + design + '; save the project first')
    context = dict((key, _blocks(designs[0], key)) for key in
        ('AnalysisSetup', 'BoundarySetup', 'MeshSetup', 'CoordinateSystems', 'MaxwellParameterSetup', 'DesignDatasets', 'LocalVariables'))
    if not context['AnalysisSetup'] or not context['BoundarySetup']:
        raise ValueError('Saved design is missing analysis/boundary setup data')
    return context


def _variables(project, design):
    result = {}
    for owner in (project, design):
        for name in owner.GetVariables():
            result[_text(name)] = _text(owner.GetVariableValue(name))
    return result


def _points(path):
    result = []
    with open(path, 'r') as stream:
        for line in csv.reader(stream):
            if not line:
                continue
            try:
                point = [float(x) for x in line]
            except ValueError:
                if not result and [x.strip() for x in line] == ['x_m', 'y_m', 'z_m']:
                    continue
                raise ValueError('points_csv must contain finite x_m,y_m,z_m in metres')
            if len(point) != 3 or not all(not math.isnan(x) and not math.isinf(x) for x in point):
                raise ValueError('Invalid observation point')
            result.append(point)
    if not result:
        raise ValueError('points_csv is empty')
    return result


def export_native(desktop, config):
    """Capture selected nominal inputs; optional fields require a saved solve."""
    required = ('project', 'design', 'solution', 'out')
    if any(not config.get(k) for k in required):
        raise ValueError('Config requires project, design, solution and out')
    reference_quantities=config.get('reference_quantities', [])
    if any(q not in ('B','H') for q in reference_quantities):
        raise ValueError('Reference quantities must be B or H')
    if reference_quantities and not config.get('points_csv'):
        raise ValueError('Reference fields require points_csv')
    project_path = os.path.abspath(config['project']); root = os.path.abspath(config['out'])
    if not os.path.isfile(project_path):
        raise ValueError('Saved project not found: ' + project_path)
    if os.path.exists(root) and os.listdir(root):
        raise ValueError('Output must be absent or empty; existing evidence is retained: ' + root)
    if not os.path.isdir(root):
        os.makedirs(root)
    for folder in ('mesh', 'native'):
        os.mkdir(os.path.join(root, folder))
    started = time.time(); before = _sha(project_path); opened = False; project = None
    info = dict(schema='aedt-native-disk-1', state='running', objects=[], materials={}, files={},
        source=dict(project=project_path, project_sha256=before, design=config['design'],
                    solution=config['solution'], aedt_version=_text(desktop.GetVersion()),exporter_sha256=_sha(__file__)),
        capture_scope='All model Solve-Inside solid volume meshes plus saved native inputs; not internal solver matrices')
    info_path = os.path.join(root, 'native_export.json')
    def register(relative, role):
        path = os.path.join(root, relative.replace('/', os.sep))
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            raise ValueError('Export did not produce a nonempty ' + role + ': ' + path)
        info['files'][relative] = dict(sha256=_sha(path), bytes=os.path.getsize(path), role=role)
        return relative
    try:
        # Reuse an already open project only when its full path is identical.
        target_name = os.path.splitext(os.path.basename(project_path))[0]
        if target_name in list(desktop.GetProjectList()):
            project = desktop.SetActiveProject(target_name)
            active_path = os.path.join(_text(project.GetPath()), _text(project.GetName()) + '.aedt')
            if os.path.normcase(os.path.abspath(active_path)) != os.path.normcase(project_path):
                raise ValueError('An open project has the same name but a different path')
        else:
            project = desktop.OpenProject(project_path); opened = True
        design = project.SetActiveDesign(config['design'])
        if _text(design.GetDesignType()) != 'Maxwell 3D' or _text(design.GetSolutionType()) != 'Magnetostatic':
            raise ValueError('Only Maxwell 3D Magnetostatic is supported by this transfer path')
        editor = design.SetActiveEditor('3D Modeler'); unit = _text(editor.GetModelUnits())
        nominal = _text(design.GetNominalVariation())
        variations=[]
        if reference_quantities:
            variations = list(design.GetModule('Solutions').GetAvailableVariations(config['solution']))
            if nominal not in variations:
                raise ValueError('The selected solution has no saved field for the current nominal variation')
        info.update(model_unit=unit, nominal_variation=nominal, available_variations=_plain(variations),
                    mode='inputs_with_reference' if reference_quantities else 'inputs_only',
                    variables=_variables(project, design), saved_context=_saved_context(project_path, config['design']))
        native_copy = os.path.join(root, 'native', 'source_project.aedt')
        shutil.copyfile(project_path, native_copy); register('native/source_project.aedt', 'saved original project inputs')
        boundaries = design.GetModule('BoundarySetup')
        excitations = list(boundaries.GetExcitations())
        if len(excitations) % 2:
            raise ValueError('Excitation name/type array is malformed')
        info['excitations'] = [dict(name=_text(excitations[i]), type=_text(excitations[i+1]),
            properties=_properties(design, 'Maxwell3D', 'BoundarySetup:' + _text(excitations[i])))
            for i in range(0, len(excitations), 2)]
        info['boundaries'] = [dict(name=_text(name), properties=_properties(design, 'Maxwell3D', 'BoundarySetup:' + _text(name)))
                              for name in boundaries.GetBoundaries()]
        info['coordinate_systems'] = _plain(editor.GetCoordinateSystems())
        info['setups'] = _plain(design.GetModule('AnalysisSetup').GetSetups())
        materials = project.GetDefinitionManager().GetManager('Material')
        reporter = design.GetModule('FieldsReporter'); sequence = 0
        for group in ('Solids', 'Sheets', 'Lines'):
            for name in editor.GetObjectsInGroup(group):
                name = _text(name); props = _properties(editor, 'Geometry3DAttributeTab', name)
                material = _text(props.get('Material', '')).strip('"')
                obj = dict(name=name, group=group, attributes=props, material=material)
                obj['object_id'] = int(editor.GetObjectIDByName(name))
                obj['vertex_ids'] = _plain(editor.GetVertexIDsFromObject(name))
                obj['cad_vertex_positions'] = [_plain(editor.GetVertexPosition(v)) for v in obj['vertex_ids']]
                obj['face_ids'] = _plain(editor.GetFaceIDs(name))
                if group == 'Solids':
                    try:
                        obj['cad_volume_model_units3'] = float(editor.GetObjectVolume(name))
                    except Exception as error:
                        obj['cad_volume_query_error'] = _text(error)
                if material and material not in info['materials']:
                    info['materials'][material] = _plain(materials.GetData(material))
                if group == 'Solids' and _bool_value(props.get('Model', True)) and _bool_value(props.get('Solve Inside', False)):
                    if not material:
                        raise ValueError('Missing material assignment: ' + name)
                    relative = 'mesh/{0:05d}.aedtplt'.format(sequence)
                    plot = 'MT_DISK_' + uuid.uuid4().hex
                    try:
                        reporter.CreateFieldPlot(['NAME:' + plot, 'SolutionName:=', config['solution'],
                            'QuantityName:=', 'Mesh', 'PlotFolder:=', plot, 'UserSpecifyName:=', 1,
                            'UserSpecifyFolder:=', 1, 'IntrinsicVar:=', '',
                            'PlotGeomInfo:=', [1, 'Volume', 'ObjList', 1, name],
                            ['NAME:PlotOnVolumeSettings', 'PlotIsoSurface:=', False, 'PointSize:=', 1,
                             'Refinement:=', 0], 'SurfaceOnly:=', False], 'Field')
                        reporter.ExportFieldPlot(plot, False, os.path.join(root, relative.replace('/', os.sep)))
                        obj['mesh_raw'] = register(relative, 'per-object volume mesh')
                    finally:
                        reporter.DeleteFieldPlot([plot])
                else:
                    obj['mesh_omitted_reason'] = 'Not a model Solve-Inside solid'
                info['objects'].append(obj); sequence += 1
                _write(info_path, info)
                print('NATIVE_EXPORT', sequence, json.dumps(name, ensure_ascii=True), obj.get('mesh_raw', obj.get('mesh_omitted_reason')))
        if not any(obj.get('mesh_raw') for obj in info['objects']):
            raise ValueError('No solved volume meshes were exported')
        if config.get('points_csv'):
            points = _points(config['points_csv'])
            shutil.copyfile(config['points_csv'], os.path.join(root, 'points.csv')); register('points.csv', 'SI observation points')
            info['points'] = 'points.csv'
            scales = dict(m=1., mm=.001, cm=.01, um=1e-6, nm=1e-9, **{'in':.0254, 'mil':2.54e-5, 'ft':.3048})
            if unit not in scales:
                raise ValueError('Unsupported observation-point model unit: ' + unit)
            with io.open(os.path.join(root, 'points.pts'), 'w', encoding='ascii') as stream:
                stream.write(u'Unit=' + unit + u'\n')
                for point in points:
                    stream.write(u' '.join(u'{0:.17g}'.format(x/scales[unit]) for x in point) + u'\n')
            register('points.pts', 'native sample points in model units')
            values = []
            for name, value in info['variables'].items():
                values.extend([name + ':=', value])
            for quantity in reference_quantities:
                reporter.CalcStack('clear'); reporter.EnterQty(quantity)
                relative = 'native/' + quantity + '.fld'
                reporter.ExportToFile(os.path.join(root, relative.replace('/', os.sep)), os.path.join(root, 'points.pts'),
                    config['solution'], values, ['NAME:ExportOption', 'IncludePtInOutput:=', True,
                        'RefCSName:=', 'Global', 'PtInSI:=', True, 'FieldInRefCS:=', True])
                info.setdefault('reference_fields', {})[quantity] = register(relative, 'native SI vector reference')
        setup = config['solution'].split(':')[0].strip()
        for method, filename in [('ExportProfile', 'profile.txt'), ('ExportConvergence', 'convergence.txt'), ('ExportMeshStats', 'mesh_stats.txt')]:
            relative = 'native/' + filename
            try:
                getattr(design, method)(setup, nominal, os.path.join(root, 'native', filename))
                register(relative, 'available native setup evidence')
                info.setdefault('native_evidence',{})[filename]=dict(state='exported',file=relative)
            except Exception as error:
                info.setdefault('native_evidence',{})[filename]=dict(state='unavailable',error=_text(error))
                partial=os.path.join(root,'native',filename)
                if os.path.isfile(partial) and os.path.getsize(partial)>0:
                    register(relative,'partial native setup evidence; see native_evidence status')
        after = _sha(project_path)
        if before != after:
            raise ValueError('Source project changed during export; bundle is not certified complete')
        info.update(state='complete', source_project_unchanged=True, elapsed_seconds=time.time()-started)
        _write(info_path, info); register('native_export.json', 'native metadata and original settings')
        manifest = dict(schema='aedt-native-disk-1', state='complete', files=info['files'])
        _write(os.path.join(root, 'manifest.json'), manifest)
        return info
    except Exception as error:
        info.update(state='failed', error=_text(error), traceback=traceback.format_exc(), elapsed_seconds=time.time()-started)
        _write(info_path, info)
        raise
    finally:
        if opened and project is not None:
            desktop.CloseProject(project.GetName())


def main(desktop=None, argument=None):
    argument = argument or (sys.argv[1] if len(sys.argv) == 2 else '')
    if not argument:
        raise ValueError('Supply the absolute export-config JSON path in Script Arguments')
    with io.open(argument.strip().strip('"'), encoding='utf-8-sig') as stream:
        config = json.load(stream)
    if desktop is None:
        import ScriptEnv
        ScriptEnv.Initialize('Ansoft.ElectronicsDesktop')
        import __main__
        desktop = __main__.oDesktop
    return export_native(desktop, config)


if __name__ == '__main__':
    main(globals().get('oDesktop'), globals().get('ScriptArgument'))
