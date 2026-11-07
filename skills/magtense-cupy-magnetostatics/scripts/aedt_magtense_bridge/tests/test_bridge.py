"""Offline tests do not call AEDT or a MagTense native solver."""
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aedtplt import read_aedtplt, tetra_volume, MeshError
from aedt_export import _coercivity, classify_material, _line_points_mm, ConversionError
from magtense_adapter import MU0, state_function_from_bh, UnsupportedPhysics, assemble_tiles
from bridge import read_maxwell_fld, check_points_in_air, main


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
