#!/usr/bin/env python3
"""Bounded, read-only QCML quantum/geometry/MD interface discovery.

This is NOT a physics solver or full data collector. No source is executed and
no model is loaded. Candidates/headers are evidence for subsequent integration,
not a claim that all required interfaces have been located or validated.
"""
import argparse,collections,csv,hashlib,io,json,os,re,stat,sys,tarfile,tempfile
from pathlib import Path
ARCHIVE_LIMIT=8*1024**2
COPY_LIMIT=4*1024**2
READ_LIMIT=32*1024**2
MAX_FILES=40000
SOURCE_CAP=256*1024
SKIP={'.git','__pycache__','site-packages','node_modules','.venv','.conda','revision_runs'}
SOURCE={'.py','.sh','.slurm','.sbatch','.yaml','.yml','.toml','.ini','.cfg'}
CATEGORIES={
 'quantum_engine':re.compile(r'pyscf|mindquantum|openfermion|qiskit|pyqpanda|pennylane|MolecularData|get_molecular_hamiltonian|get_expectation_with_grad',re.I),
 'geometry':re.compile(r'atom_coords|gto\.M\b|mole\.M\b|\.atom\s*=|\batom\s*=|read_xyz|write_xyz',re.I),
 'circuit_mapping':re.compile(r'params_name|parameter_map|paramfile|nozeroparams|VQE_param|FermionOperator|QubitOperator',re.I),
 'md_force_pes':re.compile(r'VelocityVerlet|pyscf\.md|md\.NVE|as_scanner|Epot|Ekin|Etot|CubicSpline|InterpolatedUnivariateSpline|nuc_grad',re.I),
 'optimizer':re.compile(r'scipy\.optimize|\bminimize\(|BFGS|L-BFGS|COBYLA|warm.start',re.I)}
IMPORTANT=re.compile(r'vqe|ucc|upcc|pyscf|molec|geom|force|grad|pes|aimd|dynamics|trajectory|md[_.-]',re.I)
DATA_NAME=re.compile(r'time|timing|trajectory|energy|force|\.md\.|aimd|nve',re.I)

def sha(b):return hashlib.sha256(b).hexdigest()
def dump(p,o):Path(p).write_text(json.dumps(o,indent=2,ensure_ascii=False,sort_keys=True)+'\n',encoding='utf8')
def safe_parent(parent,roots):
 """Compare canonical paths on BOTH sides before creating an output directory.

 A caller may supply a relative path or a symlink alias (for example a node's
 /tmp alias to /ram/tmp). Resolving only the output makes a lexical ancestor
 comparison miss such a protected root. Resolve every root here, rather than
 relying on collect() to have done it. strict=False permits a new output path.
 Resolution errors propagate: do not create anything if a path is ambiguous.
 """
 p=Path(parent).expanduser().resolve(strict=False)
 resolved_roots=[Path(r).expanduser().resolve(strict=False) for r in roots]
 for r in resolved_roots:
  if p==r or r in p.parents:
   raise ValueError('Output parent %s must be outside scanned root %s' % (p,r))
 p.mkdir(parents=True,exist_ok=True);return p

def collect(roots,out_parent,max_depth=7):
 roots=[Path(p).expanduser().resolve() for p in roots]
 if not roots or any(not r.is_dir() for r in roots):raise ValueError('All --root directories must exist')
 parent=safe_parent(out_parent,roots)
 out=Path(tempfile.mkdtemp(prefix='qcml_physics_interfaces_',dir=parent))
 inventory=[];hits=[];omissions=[];copied=[];headers=[];read_bytes=used=seen=0
 category_counts=collections.Counter();stopped=False
 for ri,root in enumerate(roots):
  def err(e):omissions.append({'path':str(e.filename),'reason':'WALK_ERROR','error':str(e)[:250]})
  for dp,dirs,names in os.walk(root,followlinks=False,onerror=err):
   dp=Path(dp);depth=len(dp.relative_to(root).parts)
   keep=[]
   for d in dirs:
    p=dp/d
    if p.is_symlink() or d in SKIP or d.startswith(('qcml_revision_','qcml_independent_','qcml_followup_','qcml_matrix_','qcml_lih_')):
     if len(omissions)<1000:omissions.append({'path':str(p),'reason':'EXCLUDED_DIRECTORY_POLICY_OR_SYMLINK'})
    elif depth>=max_depth:
     if len(omissions)<1000:omissions.append({'path':str(p),'reason':'DEPTH_LIMIT'})
    else:keep.append(d)
   dirs[:]=sorted(keep,key=lambda s:(not bool(IMPORTANT.search(s)),s))
   for name in sorted(names,key=lambda s:(not bool(IMPORTANT.search(s)),s)):
    seen+=1
    if seen>MAX_FILES:stopped=True;break
    p=dp/name
    try:st=p.lstat()
    except OSError as e:
     if len(omissions)<1000:omissions.append({'path':str(p),'reason':'STAT_ERROR','error':str(e)[:250]})
     continue
    if not stat.S_ISREG(st.st_mode):continue
    ext=p.suffix.lower();rel=str(p.relative_to(root));label=f'root{ri}/{rel}'
    if ext in SOURCE:
     if len(inventory)<10000:inventory.append({'path':str(p),'relative':label,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns})
     if st.st_size>SOURCE_CAP:
      if len(omissions)<1000:omissions.append({'path':str(p),'reason':'SOURCE_SIZE_CAP','bytes':st.st_size})
      continue
     if read_bytes+st.st_size>READ_LIMIT:stopped=True;break
     try:
      with p.open('rb') as f:b=f.read(SOURCE_CAP+1)
     except OSError as e:
      if len(omissions)<1000:omissions.append({'path':str(p),'reason':'READ_ERROR','error':str(e)[:250]})
      continue
     read_bytes+=len(b)
     if len(b)>SOURCE_CAP or len(b)!=st.st_size or b'\0' in b:continue
     lines=b.decode('utf8',errors='replace').splitlines();categories=[]
     for key,rx in CATEGORIES.items():
      found=[(i+1,l) for i,l in enumerate(lines) if rx.search(l)]
      if found:
       categories.append(key);category_counts[key]+=1
       if len(hits)<3000:
        hits.extend({'path':label,'line':i,'category':key,'text':l[:350]} for i,l in found[:8])
     if categories and used+len(b)<=COPY_LIMIT and len(copied)<250:
      dst=out/'source'/label;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(b)
      used+=len(b);copied.append({'path':label,'bytes':len(b),'sha256':sha(b),'categories':categories})
     elif categories:
      if len(omissions)<1000:omissions.append({'path':str(p),'reason':'SOURCE_COPY_BUDGET_OR_COUNT'})
    elif ((ext in {'.txt','.log','.out','.data','.dat','.csv','.tsv'} and DATA_NAME.search(name)) or ext=='.xyz') and len(headers)<120:
     # No numeric data file is copied in full by this branch. A bounded header/tail
     # identifies schema and code/data location; it is not a trajectory audit.
     try:
      with p.open('rb') as f:
       head=f.read(min(st.st_size,1024));f.seek(max(0,st.st_size-1024));tail=f.read(1024)
      read_bytes+=len(head)+len(tail)
      if b'\0' not in head:
       headers.append({'path':str(p),'bytes':st.st_size,
           'head':head.decode('utf8',errors='replace'),'tail':tail.decode('utf8',errors='replace'),
           'scope':'bounded byte excerpt; line alignment at tail is not guaranteed'})
     except OSError as e:
      if len(omissions)<1000:omissions.append({'path':str(p),'reason':'HEADER_READ_ERROR','error':str(e)[:200]})
   if stopped:break
  if stopped:break
 for n,o in [('source_inventory.json',inventory),('keyword_hits.json',hits),('source_copies.json',copied),
             ('data_header_candidates.json',headers),('omissions.json',omissions)]:dump(out/n,o)
 report={'status':'COLLECTED_WITH_DECLARED_LIMITS','roots':list(map(str,roots)),
         'files_visited':seen,'bytes_read':read_bytes,'copied_source_bytes':used,
         'candidate_counts':dict(category_counts),'global_limit_reached':stopped,
         'coverage':'depth/size/type bounded; not a complete directory or scientific validation',
         'originals_modified':False,'source_executed':False,'checkpoints_loaded':False,
         'archive_limit_bytes':ARCHIVE_LIMIT,'max_depth':max_depth,'file_limit':MAX_FILES}
 dump(out/'REPORT.json',report)
 archive=Path(str(out)+'.tar.gz');tmp=Path(str(archive)+'.partial')
 with tarfile.open(tmp,'w:gz',compresslevel=6) as t:t.add(out,arcname=out.name)
 if tmp.stat().st_size>ARCHIVE_LIMIT:
  tmp.unlink();raise ValueError('Archive >8 MiB; no final archive published')
 tmp.rename(archive)
 print('interface_audit = '+str(archive));print('audit_KiB = %.3f'%(archive.stat().st_size/1024))
 return archive

def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',action='append',required=True)
 a.add_argument('--out-parent',required=True);a.add_argument('--max-depth',type=int,default=7)
 x=a.parse_args()
 if not 1<=x.max_depth<=12:a.error('max-depth must be 1..12')
 collect(x.root,x.out_parent,x.max_depth)
if __name__=='__main__':
 try:main()
 except Exception as e:print('ERROR: '+str(e),file=sys.stderr);sys.exit(2)
