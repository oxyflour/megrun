"""Deliver the updated skill including its CUDA source; preserve v2/v3 ZIPs."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SKILL=ROOT/'skills/magtense-cupy-magnetostatics'
target=ROOT/'MagTense_CuPy_Skill_v4_GPU_K.zip'
assert (SKILL/'references/validation-gpu-k-20261107.json').exists()
assert '55 passed, 1 skipped' in (HERE/'tests.txt').read_text(encoding='utf-8-sig')
files=sorted(p for p in SKILL.rglob('*') if p.is_file()
       and not {'__pycache__','.pytest_cache','.git'}.intersection(p.parts)
       and p.suffix not in {'.pyc','.pyo','.log'})
kernel=SKILL/'scripts/aedt_magtense_bridge/tetra_cuda.cu'
assert kernel in files and all(p.stat().st_size<1_000_000 for p in files)
manifest=[]
with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
    for p in files:
        data=p.read_bytes();name=(Path(SKILL.name)/p.relative_to(SKILL)).as_posix()
        archive.writestr(name,data)
        manifest.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    for name in archive.namelist():
        assert name.startswith(SKILL.name+'/') and '..' not in Path(name).parts
        if name.endswith('.py'):ast.parse(archive.read(name).decode('utf-8-sig'))
with target.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
record={'package':str(target),'sha256':digest,'bytes':target.stat().st_size,'files':manifest,
        'checks':['ZIP CRC','Python syntax','CUDA source included','55 pytest passed, 1 optional Fortran test skipped',
                  'Four complete real-model K and existing-Maxwell B comparisons','Real-model CUDA CLI cold K construction']}
(HERE/'package_manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps({'package':str(target),'sha256':digest,'bytes':record['bytes'],'files':len(files)},indent=2))
