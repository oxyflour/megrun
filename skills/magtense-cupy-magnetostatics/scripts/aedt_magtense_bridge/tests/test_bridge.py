"""Offline tests do not call AEDT or a MagTense native solver."""
import json
from pathlib import Path
import sys
import types
import hashlib

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aedtplt import read_aedtplt, tetra_volume, MeshError
from aedt_export import _coercivity, classify_material, _line_points_mm, ConversionError
from magtense_adapter import MU0, state_function_from_bh, UnsupportedPhysics, assemble_tiles
from bridge import read_maxwell_fld, check_points_in_air, main


def _native_disk_fixture(root,*,material=None,excitation=None,orientation='Global'):
    """Synthetic transfer fixture: known 1 mm tetrahedron, not Maxwell evidence."""
    root.mkdir();(root/'mesh').mkdir();(root/'native').mkdir()
    synthetic_aedtplt(root/'mesh/00000.aedtplt')
    (root/'native/source_project.aedt').write_text('Synthetic native-transfer fixture; no Ansys solution')
    (root/'points.csv').write_text('x_m,y_m,z_m\n0.002,0.002,0.002\n-0.002,0,0\n')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    info=dict(schema='aedt-native-disk-1',state='complete',source_project_unchanged=True,
        model_unit='mm',variables={},excitations=[] if excitation is None else [excitation],boundaries=[],
        saved_context={'AnalysisSetup':['synthetic setup'], 'BoundarySetup':['synthetic boundaries']},
        source=dict(project='synthetic fixture',project_sha256=sha(root/'native/source_project.aedt'),
            design='fixture',solution='Reference : LastAdaptive',aedt_version='18.2-fixture'),
        capture_scope='Synthetic input transfer, not native API/Maxwell accuracy validation',points='points.csv',
        materials={'Fixture':material or ['NAME:Fixture','permeability:=','2']},
        objects=[dict(name='tetra',group='Solids',material='Fixture',mesh_raw='mesh/00000.aedtplt',
            attributes={'Model':'true','Solve Inside':'true','Orientation':orientation},
            cad_vertex_positions=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]])])
    (root/'native_export.json').write_text(json.dumps(info),encoding='utf-8')
    _native_manifest(root)
    return info


def _native_manifest(root):
    files={str(p.relative_to(root)).replace('\\','/'):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
        for p in root.rglob('*') if p.is_file() and p.name!='manifest.json'}
    (root/'manifest.json').write_text(json.dumps(dict(schema='aedt-native-disk-1',state='complete',files=files)),encoding='utf-8')


def synthetic_aedtplt(path, tet10=True, element_type=4):
    # TET10 nodes: corners at [0,2,5,9]; mid-side nodes deliberately not corners.
    nodes = np.array([
        [0,0,0],      # corner 0
        [0.5,0,0],    # midpoint
        [1,0,0],      # corner 1
        [0.5,0.5,0],
        [0,0.5,0],
        [0,1,0],      # corner 2
        [0,0,0.5],
        [0.5,0,0.5],
        [0,0.5,0.5],
        [0,0,1],      # corner 3
    ],float)
    if tet10:
        ids=list(range(1,11))
    else:
        ids=[1,3,6,10]
    ints=[len(nodes),1,element_type,0,0,0,len(ids),*ids]
    string='''# aedtplt synthetic fixture\n$begin Drawing_0\nHasCurvElem(0)\nNodes('''
    string+=','.join(map(str,nodes.flatten()))+')\nElements('
    string+=','.join(map(str,ints))+')\n$end Drawing_0\n'
    path.write_text(string)


@pytest.mark.parametrize('tet10',[False,True])
def test_parser_tetra_corner_selection(tmp_path,tet10):
    p=tmp_path/'tiny.aedtplt'
    synthetic_aedtplt(p,tet10=tet10)
    v,stat=read_aedtplt(p,1e-3)
    assert v.shape==(1,4,3)
    np.testing.assert_allclose(np.sort(v[0],axis=0),
                               np.sort(np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])*1e-3,axis=0))
    np.testing.assert_allclose(np.abs(tetra_volume(v))[0],1e-9/6)
    assert stat['tet10']==int(tet10)
    assert stat['tet4']==int(not tet10)


def test_native_bundle_import_and_zero_field_solve_without_ansys_or_magtense(tmp_path,monkeypatch):
    """No Maxwell/PyAEDT/native DLL; exact zero-source field is independently zero."""
    for name in ('pyaedt','ansys','ScriptEnv','magtense'):
        monkeypatch.setitem(sys.modules,name,None)
    bundle=tmp_path/'bundle';_native_disk_fixture(bundle);job=tmp_path/'job'
    assert main(['import-native','--bundle',str(bundle),'--out',str(job)])==0
    model=json.loads((job/'model.json').read_text())
    assert model['total_magnetic_tetra']==1
    assert model['objects'][0]['properties']=={'kind':'linear','mu_r':2.}
    vertices=np.load(job/'mesh/00000.npz')['tetra_m']
    np.testing.assert_allclose(vertices[0],np.eye(4,3,k=-1)*1e-3,rtol=0,atol=0)
    np.testing.assert_allclose(tetra_volume(vertices),[1e-9/6])
    assert main(['solve-fem-bem','--out',str(job)])==0
    info=json.loads((job/'fem_bem_solve_info.json').read_text())
    assert info['passed'] and info['solver']['residual']==0
    prediction=np.loadtxt(job/'fem_bem_B.csv',delimiter=',',skiprows=1)
    np.testing.assert_array_equal(prediction[:,3:],0)
    original=(job/'fem_bem_solution.npz').read_bytes()
    assert main(['solve-fem-bem','--out',str(job)])==2
    assert (job/'fem_bem_solution.npz').read_bytes()==original


@pytest.mark.parametrize('member',['mesh/00000.npz','points.csv'])
def test_imported_job_detects_transfer_corruption(tmp_path,member):
    from aedt_export import import_native_bundle
    from bridge import read_model
    bundle=tmp_path/'bundle';_native_disk_fixture(bundle);job=tmp_path/'job'
    import_native_bundle(bundle,job)
    path=job/member;path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='SHA-256 mismatch'):read_model(job)


@pytest.mark.parametrize('node_scale',[1.,.001])
def test_native_volume_verifies_units_without_cad_vertices(tmp_path,node_scale):
    from aedt_export import import_native_bundle
    bundle=tmp_path/'bundle';info=_native_disk_fixture(bundle)
    obj=info['objects'][0];obj['cad_vertex_positions']=[];obj['cad_volume_model_units3']=1/6
    (bundle/'native_export.json').write_text(json.dumps(info))
    nodes=np.r_[np.zeros((1,3)),np.eye(3)]*node_scale
    (bundle/'mesh/00000.aedtplt').write_text('$begin Drawing_0\nNodes('+','.join(map(str,nodes.ravel()))+
        ')\nElements(4,1,4,0,0,0,4,1,2,3,4)\n$end Drawing_0\n')
    _native_manifest(bundle)
    model=import_native_bundle(bundle,tmp_path/'job')
    np.testing.assert_allclose(model['objects'][0]['mesh_stats']['volume_m3'],1e-9/6)
    assert model['objects'][0]['mesh_stats']['cad_units_verification']=='native CAD volume'


@pytest.mark.parametrize('mode',['corrupt_mesh','missing_file','traversal','incomplete'])
def test_native_bundle_rejects_corruption_or_incomplete_archive_before_output(tmp_path,mode):
    from aedt_export import import_native_bundle
    bundle=tmp_path/'bundle';_native_disk_fixture(bundle);job=tmp_path/'job'
    path=bundle/'manifest.json';manifest=json.loads(path.read_text())
    if mode=='corrupt_mesh':
        mesh=bundle/'mesh/00000.aedtplt';mesh.write_text(mesh.read_text().replace('0.5','0.6',1))
    elif mode=='missing_file':(bundle/'points.csv').unlink()
    elif mode=='traversal':manifest['files']['../outside']={'bytes':1,'sha256':'0'*64}
    else:manifest['state']='failed'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ConversionError):import_native_bundle(bundle,job)
    assert not job.exists()


@pytest.mark.parametrize('value',['1A','-1mA','unknown_current','0A * unknown'])
def test_native_bundle_preserves_but_refuses_unmodeled_current(tmp_path,value):
    from aedt_export import import_native_bundle
    bundle=tmp_path/'bundle'
    _native_disk_fixture(bundle,excitation={'name':'coil','type':'Current','properties':{'Current':value}})
    with pytest.raises(ConversionError,match='current'):
        import_native_bundle(bundle,tmp_path/'job')
    assert not (tmp_path/'job').exists()


def test_native_bh_units_signed_coercivity_and_coordinate_system_guard(tmp_path):
    from aedt_export import _native_material,import_native_bundle
    bh=['NAME:Steel',['NAME:permeability','property_type:=','nonlinear','HUnit:=','ka_per_meter',
        'BUnit:=','mt','IsTemperatureDependent:=',False,['NAME:BHCoordinates',
            ['NAME:Point',0,0],['NAME:Point',1,1000],['NAME:Point',2,1500]]]]
    props=classify_material(_native_material('Steel',bh))
    assert props['bh_H_Apm_B_T']==[[0,0],[1000,1],[2000,1.5]]
    permanent=['NAME:PM','permeability:=','1.05',['NAME:magnetic_coercivity','Magnitude:=','-900000A_per_meter',
        'DirComp1:=','0','DirComp2:=','0','DirComp3:=','1']]
    props=classify_material(_native_material('PM',permanent))
    np.testing.assert_allclose(props['br_T'],MU0*1.05*900000)
    assert props['direction']==[0,0,1]
    bundle=tmp_path/'bundle';_native_disk_fixture(bundle,material=permanent,orientation='LocalCS')
    with pytest.raises(ConversionError,match='global-direction'):
        import_native_bundle(bundle,tmp_path/'rejected')
    overrides=tmp_path/'overrides.json';overrides.write_text(json.dumps({'objects':{'tetra':props}}))
    assert import_native_bundle(bundle,tmp_path/'job',overrides_path=overrides)['objects'][0]['properties']==props


def test_native_boolean_metadata_and_saved_context_are_python27_compatible(tmp_path):
    from aedt_export_native import _bool_value,_saved_context,_write,_plain
    assert _bool_value('true') and not _bool_value('False')
    with pytest.raises(ValueError):_bool_value('parameter_expression')
    project=tmp_path/'saved.aedt'
    project.write_bytes(b"$begin 'Maxwell3DModel'\nName='case'\n$begin 'AnalysisSetup'\nSetting='x'\n$end 'AnalysisSetup'\n"
        b"$begin 'BoundarySetup'\n$end 'BoundarySetup'\n$end 'Maxwell3DModel'\n\xff\x00")
    context=_saved_context(str(project),'case')
    assert "Setting='x'" in context['AnalysisSetup'][0]
    path=tmp_path/'values.json';_write(str(path),_plain({'b':[True,False], 'text':'磁体'}))
    assert json.loads(path.read_text(encoding='utf-8'))=={'b':[True,False], 'text':'磁体'}


def test_native_script_arguments_read_utf8_config_without_external_packages(tmp_path,monkeypatch):
    import aedt_export_native as native
    config=tmp_path/'导出配置.json';data={'design':'磁体','reference_quantities':[]}
    config.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8-sig')
    desktop=object();captured=[]
    monkeypatch.setattr(native,'export_native',lambda d,c:captured.append((d,c)) or c)
    assert native.main(desktop,'"'+str(config)+'"')==data
    assert captured==[(desktop,data)]


@pytest.mark.parametrize('request_reference',[False,True])
def test_native_mesh_only_export_needs_no_field_solution(tmp_path,request_reference):
    """Native API contract stub: generated volume mesh exists, fields do not."""
    from aedt_export_native import export_native
    from aedt_export import import_native_bundle
    project_file=tmp_path/'fixture.aedt'
    project_file.write_text("$begin 'Maxwell3DModel'\nName='fixture'\n$begin 'AnalysisSetup'\n$end 'AnalysisSetup'\n"
        "$begin 'BoundarySetup'\n$end 'BoundarySetup'\n$end 'Maxwell3DModel'\n")
    original=project_file.read_bytes();points=tmp_path/'points.csv'
    points.write_text('x_m,y_m,z_m\n0.002,0.002,0.002\n')
    calls=[];attrs={'Model':'true','Solve Inside':'true','Material':'"Fixture"','Orientation':'Global'}
    editor=types.SimpleNamespace(GetModelUnits=lambda:'mm',GetCoordinateSystems=lambda:['Global'],
        GetObjectsInGroup=lambda group:['tetra'] if group=='Solids' else [],GetObjectIDByName=lambda n:1,
        GetProperties=lambda *a:list(attrs),GetPropertyValue=lambda tab,n,key:attrs[key],
        GetVertexIDsFromObject=lambda n:[0,1,2,3],GetVertexPosition=lambda n:np.r_[np.zeros((1,3)),np.eye(3)][n].tolist(),
        GetFaceIDs=lambda n:[1,2,3,4],GetObjectVolume=lambda n:1/6)
    reporter=types.SimpleNamespace(CreateFieldPlot=lambda *a:calls.append('create'),
        ExportFieldPlot=lambda name,flag,path:synthetic_aedtplt(Path(path),tet10=False),
        DeleteFieldPlot=lambda *a:calls.append('delete'))
    modules={'FieldsReporter':reporter,'BoundarySetup':types.SimpleNamespace(GetExcitations=lambda:[],GetBoundaries=lambda:[]),
        'AnalysisSetup':types.SimpleNamespace(GetSetups=lambda:['Reference']),
        'Solutions':types.SimpleNamespace(GetAvailableVariations=lambda s:calls.append('field_lookup') or [])}
    def unavailable(*args):raise RuntimeError('No field solution exists')
    design=types.SimpleNamespace(GetDesignType=lambda:'Maxwell 3D',GetSolutionType=lambda:'Magnetostatic',
        SetActiveEditor=lambda n:editor,GetNominalVariation=lambda:'',GetVariables=lambda:[],GetModule=lambda n:modules[n],
        ExportProfile=unavailable,ExportConvergence=unavailable,ExportMeshStats=unavailable)
    project=types.SimpleNamespace(GetName=lambda:'fixture',SetActiveDesign=lambda n:design,GetVariables=lambda:[],
        GetDefinitionManager=lambda:types.SimpleNamespace(GetManager=lambda n:types.SimpleNamespace(GetData=lambda m:['NAME:Fixture','permeability:=','2'])))
    desktop=types.SimpleNamespace(GetVersion=lambda:'18.x-contract-fixture',GetProjectList=lambda:[],
        OpenProject=lambda p:project,CloseProject=lambda n:calls.append('close'))
    config=dict(project=str(project_file),design='fixture',solution='Reference : LastAdaptive',
        out=str(tmp_path/'export'),points_csv=str(points),reference_quantities=['B'] if request_reference else [])
    if request_reference:
        with pytest.raises(ValueError,match='no saved field'):export_native(desktop,config)
        assert not (tmp_path/'export/manifest.json').exists()
    else:
        info=export_native(desktop,config)
        assert info['mode']=='inputs_only' and not info.get('reference_fields')
        assert all(v['state']=='unavailable' for v in info['native_evidence'].values())
        assert calls==['create','delete','close']
        assert import_native_bundle(tmp_path/'export',tmp_path/'job')['total_magnetic_tetra']==1
    assert project_file.read_bytes()==original


def test_parser_rejects_surface(tmp_path):
    p=tmp_path/'surface.aedtplt'
    synthetic_aedtplt(p,tet10=False,element_type=2)
    with pytest.raises(MeshError,match='No tetrahedral'):
        read_aedtplt(p,1e-3)


def test_negative_winding_flipped(tmp_path):
    p=tmp_path/'tiny.aedtplt'
    synthetic_aedtplt(p,tet10=True)
    s=p.read_text()
    s=s.replace('4,0,0,0,10,1,2,3,4,5,6,7,8,9,10',
                '4,0,0,0,10,3,2,1,4,5,6,7,8,9,10')
    p.write_text(s)
    v,stats=read_aedtplt(p,1e-3)
    assert stats['orientation_flipped']==1
    assert tetra_volume(v)[0] > 0



def test_line_points_unit():
    pts=_line_points_mm('0,0,2:10,0,2:11')
    assert pts.shape==(11,3)
    np.testing.assert_allclose(pts[-1],[.01,0,.002])


class FakeMat:
    def __init__(self, mur, hc=0, direction=(1,0,0), nonlinear=False):
        self.name='NdFeB'
        self.permeability=types.SimpleNamespace(type='nonlinear' if nonlinear else 'simple',
                            value=[[0,0],[100,0.1],[1000,0.3]] if nonlinear else str(mur),
                            hunit='A_per_meter',bunit='tesla',is_temperature_dependent=False)
        self._props={'magnetic_coercivity': {
            'Magnitude': f'{hc}A_per_meter',
            'DirComp1':direction[0], 'DirComp2':direction[1], 'DirComp3':direction[2]}}


def test_material_skip_and_coercivity_sign():
    assert classify_material(FakeMat(1.0,0))['kind']=='skip'
    pm=classify_material(FakeMat(1.05,-900000,direction=(0,0,1)))
    assert pm['kind']=='permanent'
    np.testing.assert_allclose(pm['br_T'],MU0*1.05*900000)
    assert pm['direction']==[0,0,1]
    assert classify_material(FakeMat(250))['kind']=='linear'
    assert classify_material(FakeMat(1,nonlinear=True))['kind']=='nonlinear'


def test_bh_state_function_conversion():
    sf=state_function_from_bh([[0,0],[100,0.1],[1000,0.7]])
    np.testing.assert_allclose(sf[0],[3,300])
    np.testing.assert_allclose(sf[2],[100,0.1/MU0-100])
    assert sf.shape==(4,2)
    with pytest.raises(UnsupportedPhysics):
        state_function_from_bh([[0,0],[100,0.1],[100,0.2]])


def test_fld_coordinates(tmp_path):
    p=tmp_path/'field.fld'
    p.write_text('Unit=mm\n0 0 0 0.5 0 0\n1 0 0 0.4 0 0\n')
    B,scale=read_maxwell_fld(p,np.array([[0,0,0],[1e-3,0,0]]))
    assert scale==1e-3
    np.testing.assert_allclose(B[:,0],[0.5,0.4])
    with pytest.raises(ValueError,match='do not align'):
        read_maxwell_fld(p,np.array([[0,0,0],[4e-3,0,0]]))


def test_end_to_end_offline_demo_and_sample_in_air(tmp_path):
    assert main(['demo','--out',str(tmp_path)])==0
    assert main(['inspect','--out',str(tmp_path)])==0
    assert main(['solve','--out',str(tmp_path),'--dry-run'])==0
    model=json.loads((tmp_path/'model.json').read_text())
    assert model['total_magnetic_tetra']==6
    assert np.isclose(model['objects'][0]['mesh_stats']['volume_m3'],1e-9)
    with pytest.raises(ValueError,match='on/in magnetic'):
        check_points_in_air(tmp_path,model,np.array([[0.5e-3]*3]))


def test_tile_assembly_fake_magtense_monkeypatch(tmp_path,monkeypatch):
    class FakeTiles:
        def __init__(self,**kwargs):
            self.n=kwargs['n']
            self.vertices=kwargs['vertices']
            self.M_rem=kwargs['M_rem']
            self.u_ea=kwargs['easy_axis']
            self.magnet_type=None
            self.incl_it=None
            self.stfcn_index=None
    fake=types.ModuleType('magtense')
    sub=types.ModuleType('magtense.magstatics')
    sub.Tiles=FakeTiles
    monkeypatch.setitem(sys.modules,'magtense',fake)
    monkeypatch.setitem(sys.modules,'magtense.magstatics',sub)
    assert main(['demo','--out',str(tmp_path)])==0
    m=json.loads((tmp_path/'model.json').read_text())
    t,bh,info=assemble_tiles(m['objects'],tmp_path,max_tiles=100)
    assert t.n==6 and t.vertices.shape==(6,3,4)
    assert np.all(t.magnet_type==1) and np.all(t.incl_it==1)
    assert bh is None
    assert info['tiles']==6


def test_nonlin_fortran_state_function_is_forwarded(monkeypatch):
    """Verify the adapter calls the low-level wrapper instead of packaged Fe curve."""
    from magtense_adapter import solve as solve_mt
    captured={}
    class F:
        def runsimulation(self,**kw):
            captured.update(kw)
            return np.full((2,3),100.0), np.zeros((3,3)), np.full(3,1e-6)
    lib=types.ModuleType('magtense.lib')
    lib.magtensesource=types.SimpleNamespace(fortrantopythonio=F())
    mag=types.ModuleType('magtense.magstatics')
    mag.run_simulation=lambda *a,**k: (_ for _ in ()).throw(AssertionError('must not use packaged curve'))
    fake_pkg=types.ModuleType('magtense')
    fake_pkg.__path__=[]
    fake_pkg.lib=lib
    fake_pkg.magstatics=mag
    monkeypatch.setitem(sys.modules,'magtense',fake_pkg)
    monkeypatch.setitem(sys.modules,'magtense.lib',lib)
    monkeypatch.setitem(sys.modules,'magtense.magstatics',mag)
    class T:
        n=3
        center_pos=np.zeros((3,3)); dev_center=np.zeros((3,3)); size=np.zeros((3,3))
        vertices=np.zeros((3,3,4)); M=np.zeros((3,3))
        u_ea=np.zeros((3,3)); u_oa1=np.zeros((3,3)); u_oa2=np.zeros((3,3))
        mu_r_ea=np.ones(3); mu_r_oa=np.ones(3); M_rem=np.zeros(3)
        tile_type=np.ones(3,dtype=int)*5; offset=np.zeros((3,3)); rot=np.zeros((3,3))
        color=np.zeros((3,3)); magnet_type=np.ones(3,dtype=int)*2
        stfcn_index=np.ones(3,dtype=int); incl_it=np.ones(3,dtype=int)
        use_sym=np.zeros(3,dtype=int); sym_op=np.ones((3,3)); M_rel=np.zeros(3)
    t=T()
    B,diagnostics=solve_mt(t, np.zeros((2,3)),
        np.array([[0,0],[100,0.1],[1000,0.3]]))
    assert captured['n_statefcn']==1
    assert captured['data_statefcn'].shape==(4,2)
    assert captured['data_statefcn'][2,0]==100
    np.testing.assert_allclose(B,np.full((2,3),100*MU0))
    assert diagnostics['max_final_M_rel']==1e-6


def test_metrics_pipeline_with_synthetic_reference(tmp_path):
    assert main(['demo','--out',str(tmp_path)])==0
    m=json.loads((tmp_path/'model.json').read_text())
    p=np.loadtxt(tmp_path/'points.csv', delimiter=',', skiprows=1)
    ref=np.repeat([[0.001,0,0]],len(p),axis=0)
    pred=ref+np.repeat([[0.0001,0,0]],len(p),axis=0)
    np.savetxt(tmp_path/'magtense_B.csv',np.column_stack((p,pred)),
               delimiter=',',header='x_m,y_m,z_m,Bx_T,By_T,Bz_T',comments='')
    np.savetxt(tmp_path/'reference.csv',np.column_stack((p,ref)),
               delimiter=',',header='x_m,y_m,z_m,Bx_T,By_T,Bz_T',comments='')
    assert main(['compare','--out',str(tmp_path),'--maxwell-csv',str(tmp_path/'reference.csv')])==0
    result=json.loads((tmp_path/'metrics.json').read_text())
    np.testing.assert_allclose(result['rms_vector_relative_error_percent'],10)
    np.testing.assert_allclose(result['rms_vector_abs_error_mT'],0.1)
