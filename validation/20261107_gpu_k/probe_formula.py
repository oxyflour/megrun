"""Independent surface-charge triangle formula, checked against native TET4."""
import sys
from pathlib import Path
import numpy as np
from magtense.magstatics import Tiles, get_demag_tensor


def tensor(v, p):
    result = np.zeros((len(p), 3, 3))
    for opposite in range(4):
        face = v[[i for i in range(4) if i != opposite]].copy()
        n = np.cross(face[1]-face[0], face[2]-face[0])
        if np.dot(n, v[opposite]-face[0]) > 0:
            face[[1,2]] = face[[2,1]]
            n = -n
        n /= np.linalg.norm(n)
        R = face[None,:,:]-p[:,None,:]
        norms = np.linalg.norm(R, axis=2)
        numerator = np.einsum('pi,pi->p', R[:,0], np.cross(R[:,1], R[:,2]))
        denominator = np.prod(norms,axis=1)
        for i in range(3):
            denominator += np.einsum('pi,pi->p',R[:,i],R[:,(i+1)%3])*norms[:,(i+2)%3]
        omega = 2*np.arctan2(numerator,denominator)
        field = -omega[:,None]*n
        for i in range(3):
            edge = face[(i+1)%3]-face[i]
            length = np.linalg.norm(edge)
            outward = np.cross(edge/length, n)
            s = norms[:,i]+norms[:,(i+1)%3]
            integral = np.log1p(2*length/(s-length))
            field += integral[:,None]*outward
        result += field[:,:,None]*n[None,None,:]/(4*np.pi)
    return result


if __name__ == '__main__':
    rng = np.random.default_rng(127)
    v = np.array([[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.]])*.001
    points = np.r_[v.mean(0)[None],rng.normal(size=(300,3))*.002]
    tile = Tiles(n=1,vertices=np.asfortranarray(v.T[None]),tile_type=5,
                 M_rem=0,easy_axis=[1,0,0],mu_r_ea=1,mu_r_oa=1)
    native = get_demag_tensor(tile,points)[0]
    own = tensor(v,points)
    print('self native',native[0]);print('self surface',own[0])
    print('relative',np.linalg.norm(own-native)/np.linalg.norm(native),'maxabs',np.max(abs(own-native)))
