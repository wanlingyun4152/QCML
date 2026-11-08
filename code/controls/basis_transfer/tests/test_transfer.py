"""Synthetic functional tests. No supplied VQE arrays, no physical performance claims."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv,copy,hashlib,json,tempfile,unittest,os,subprocess,types
from unittest.mock import patch
import numpy as np
import torch
import data as dat
import models as mod
import train as tr
import transfer as main

torch.set_num_threads(1)

def fixture(root):
    root=Path(root);root.mkdir()
    sp=json.loads((Path(__file__).parent/'split_fixture.json').read_text())
    files={};basis=['sto-3g','6-31g*','cc-pvdz']
    for bi,(norb,nparam) in enumerate([(6,44),(16,434),(19,629)]):
        d=root/('basis_%d'%bi);d.mkdir()
        for split in ['train','val','test']:
            rv=sorted(float(k) for k,v in sp['assignment'].items() if v==split)
            rr=np.array(rv);n=len(rr)
            ids=np.array(['LiH|%s|R=%s'%(basis[bi],next(k for k in sp['assignment'] if float(k)==r)) for r in rv])
            p=d/(split+'.npz')
            if split=='test':p.write_bytes(b'POISON: NEVER READ THESE TEST LABELS')
            else:
                x=np.column_stack([rr,np.full(n,norb),np.full(n,4),np.full(n,nparam),-.4+.03*rr,.1-.02*rr])
                y=.03*np.sin(rr[:,None]*(np.arange(nparam)[None,:]+1)/nparam)+.005*bi
                np.savez_compressed(p,X_raw=x,theta=y,sample_id=ids,R_angstrom=rr,feature_names=np.array(mod.FEATURES))
            files[str(p.relative_to(root))]=dat.sha(p)
            q=d/(split+'_index.tsv')
            with q.open('w',newline='') as f:
                w=csv.writer(f,delimiter='\t');w.writerow(['sample_id','R_angstrom','point_index','raw_point_directory','label_sha256'])
                for i,r in enumerate(rv):w.writerow([ids[i],r,bi*202+sp['grid'].index(next(k for k in sp['grid'] if float(k)==r)),'synthetic-only','testhash'])
            files[str(q.relative_to(root))]=dat.sha(q)
        mp=d/'parameter_map.json';mp.write_text(json.dumps({'synthetic_parameter_count':nparam}));files[str(mp.relative_to(root))]=dat.sha(mp)
    s=root/'split.json';s.write_text(json.dumps(sp));files['split.json']=dat.sha(s)
    m={'status':'READY_FOR_GAUGE_REVIEW','coordinate_protocol':{'version':'stage4_two_chart_regauge_v1_1_pullback'},'source_results':{str(i):{} for i in range(606)},'feature_names':mod.FEATURES,'files':files}
    q=root/'manifest.json';q.write_text(json.dumps(m));return {'manifest_sha256':dat.sha(q),'files':files}

class Safety(unittest.TestCase):
    def setUp(self):self.t=tempfile.TemporaryDirectory();self.p=Path(self.t.name)
    def tearDown(self):self.t.cleanup()
    def test_descendant_blocked(self):
        s=self.p/'a';s.mkdir()
        with self.assertRaises(ValueError):dat.safe_parent(s/'out',[s])
        self.assertFalse((s/'out').exists())
    def test_sibling_resolved(self):
        s=self.p/'a';s.mkdir();self.assertEqual(dat.safe_parent(self.p/'ab',[s]),(self.p/'ab').resolve())
    def test_alias_blocked(self):
        s=self.p/'a';s.mkdir();link=self.p/'alias';link.symlink_to(s,target_is_directory=True)
        with self.assertRaises(ValueError):dat.safe_parent(link/'out',[s])
    def test_source_alias_blocked(self):
        s=self.p/'a';s.mkdir();link=self.p/'alias';link.symlink_to(s,target_is_directory=True)
        with self.assertRaises(ValueError):dat.safe_parent(s/'out',[link])
    def test_same_path_blocked(self):
        with self.assertRaises(ValueError):dat.safe_parent(self.p,[self.p])
    def test_no_overwrite_run(self):
        with self.assertRaises(FileExistsError):tr.train_run(self.p,'mlp',0,{}, {},1)
    def test_two_protected_roots(self):
        a=self.p/'a';a.mkdir();b=self.p/'b';b.mkdir()
        with self.assertRaises(ValueError):dat.safe_parent(b/'o',[a,b])

class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t=tempfile.TemporaryDirectory();cls.root=Path(cls.t.name)/'dataset';cls.lock=fixture(cls.root)
    @classmethod
    def tearDownClass(cls):cls.t.cleanup()
    def test_shapes_and_split(self):
        _,s,d=dat.load_data(self.root,self.lock)
        self.assertEqual(d[2]['train']['theta'].shape,(142,629));self.assertEqual(d[1]['val']['theta'].shape,(30,434))
        self.assertEqual(s['counts']['test'],30)
    def test_no_test_numpy_load(self):
        orig=np.load;called=[]
        def guard(p,*a,**kw):
            called.append(str(p));self.assertFalse(Path(p).name=='test.npz');return orig(p,*a,**kw)
        with patch.object(np,'load',side_effect=guard):dat.load_data(self.root,self.lock)
        self.assertEqual(len(called),6)
    def test_no_test_hash(self):
        orig=dat.sha;called=[]
        def guard(p):called.append(str(p));self.assertNotEqual(Path(p).name,'test.npz');return orig(p)
        with patch.object(dat,'sha',side_effect=guard):dat.load_data(self.root,self.lock)
    def test_test_split_explicit_rejected(self):
        with self.assertRaises(ValueError):dat.load_split(self.root,0,'test',{})
    def test_lock_mismatch(self):
        bad=dict(self.lock,manifest_sha256='bad')
        with self.assertRaises(ValueError):dat.load_data(self.root,bad)
    def test_bad_features(self):
        with self.assertRaises(ValueError):mod.features(np.ones((2,7)))
    def test_nonfinite_features(self):
        with self.assertRaises(ValueError):mod.features(np.full((2,6),np.nan))
    def test_fixed_scaling(self):
        x=np.array([[1.8,20,4,640,1,-1.]])
        self.assertTrue(np.array_equal(mod.features(x),[[0,1,1,1,1,-1]]))
    def test_scaling_does_not_mutate(self):
        x=np.ones((3,6));old=x.copy();mod.features(x);self.assertTrue(np.array_equal(x,old))
    def test_nested_subsets(self):
        _,sp,d=dat.load_data(self.root,self.lock);previous=set()
        for n in main.SIZES:
            small=dat.subset(d[2]['train'],sp['nested_training_sets'][str(n)])
            ids=set(small['sample_id']);self.assertEqual(len(ids),n);self.assertTrue(previous<=ids);previous=ids

class ModelTests(unittest.TestCase):
    def test_outputs(self):
        for f in ['transformer','mlp']:
            for n in [44,434,629]:self.assertEqual(mod.initialize(f,n,0)(torch.randn(3,6)).shape,(3,n))
    def test_parameter_scale(self):
        for n in [44,434,629]:
            a=sum(x.numel() for x in mod.initialize('transformer',n,0).parameters());b=sum(x.numel() for x in mod.initialize('mlp',n,0).parameters())
            self.assertLess(abs(a-b)/a,.01)
    def test_independence(self):
        x=np.random.RandomState(0).normal(size=(7,6)).astype('float32')
        for f in ['transformer','mlp']:self.assertTrue(mod.invariance(mod.initialize(f,629,0),x,x)['passed'])
    def test_same_initial_heads(self):
        for f in ['transformer','mlp']:
            src=mod.initialize(f,44,9).state_dict()
            a=mod.initialize(f,434,0);b=mod.initialize(f,434,0,src)
            self.assertEqual(mod.state_hash(a.head.state_dict()),mod.state_hash(b.head.state_dict()))
            self.assertNotEqual(mod.state_hash(a.backbone.state_dict()),mod.state_hash(b.backbone.state_dict()))
    def test_transferred_backbone(self):
        src=mod.initialize('transformer',44,2)
        dst=mod.initialize('transformer',629,0,src.state_dict())
        self.assertEqual(mod.state_hash(src.backbone.state_dict()),mod.state_hash(dst.backbone.state_dict()))
    def test_wrong_model_rejected(self):
        with self.assertRaises(ValueError):mod.initialize('gnn',44,0)
    def test_separate_layer_initialization(self):
        m=mod.initialize('transformer',44,0)
        self.assertFalse(torch.equal(m.backbone.layers[0].linear1.weight,m.backbone.layers[1].linear1.weight))
    def test_reproducible_order(self):self.assertTrue(np.array_equal(tr.order_for(142,1,20),tr.order_for(142,1,20)))
    def test_different_epochs_order(self):self.assertFalse(np.array_equal(tr.order_for(142,1,20),tr.order_for(142,1,21)))
    def test_metrics(self):
        x=tr.metrics(np.array([[1.,2.]]),np.zeros((1,2)))
        self.assertAlmostEqual(x['MSE'],2.5)

class FlowTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.p=Path(self.t.name);self.root=self.p/'data';self.lock=fixture(self.root)
    def tearDown(self):self.t.cleanup()
    def test_complete_synthetic_training_pair(self):
        _,sp,d=dat.load_data(self.root,self.lock)
        for f in ['transformer','mlp']:
            source=self.p/(f+'_source')
            a=tr.train_run(source,f,0,d[0]['train'],d[0]['val'],2)
            target=dat.subset(d[1]['train'],sp['nested_training_sets']['8'])
            s=tr.train_run(self.p/(f+'_scratch'),f,0,target,d[1]['val'],2)
            t=tr.train_run(self.p/(f+'_transfer'),f,0,target,d[1]['val'],2,source=source/'best.pt')
            self.assertEqual(s['train_order_sha256'],t['train_order_sha256'])
            self.assertEqual(s['initial_head_sha256'],t['initial_head_sha256'])
            self.assertTrue(a['checkpoint_reload_passed'] and t['invariance']['passed'])
            self.assertFalse(t['ML_final_test_evaluated'])
    def test_prepare_and_full_plan(self):
        with patch.object(main,'read',side_effect=lambda p:self.lock if str(p).endswith('dataset_lock.json') else dat.read(p)):
            suite=main.prepare(self.root,self.p/'output',epochs=1)
        plan=dat.read(suite/'plan.json');self.assertEqual(len(list(main.expected_runs(plan))),150)
        self.assertEqual(len(plan['bundles']),6)
        main.check_code(plan)
    def test_incomplete_collect(self):
        with patch.object(main,'read',side_effect=lambda p:self.lock if str(p).endswith('dataset_lock.json') else dat.read(p)):
            suite=main.prepare(self.root,self.p/'output',epochs=1)
        dest=main.collect(suite);self.assertLess(dest.stat().st_size,12*1024**2)
    def test_archive_hard_cap(self):
        with patch.object(main,'read',side_effect=lambda p:self.lock if str(p).endswith('dataset_lock.json') else dat.read(p)):
            suite=main.prepare(self.root,self.p/'output',epochs=1)
        with patch.object(main,'ARCHIVE_CAP',1):
            with self.assertRaises(RuntimeError):main.collect(suite)
        self.assertFalse(list((self.p/'output').glob('*.tar.gz')))
    def test_queue_array_count(self):
        v=types.SimpleNamespace(returncode=0,stdout='101_0\n101_1\n102\n',stderr='')
        with patch.object(main.subprocess,'run',return_value=v):self.assertEqual(main.queue_count(),3)
    def test_unexpanded_queue_rejected(self):
        v=types.SimpleNamespace(returncode=0,stdout='101_[0-29]\n',stderr='')
        with patch.object(main.subprocess,'run',return_value=v):
            with self.assertRaises(ValueError):main.queue_count()
    def test_no_slots_no_prepare(self):
        with patch.object(main,'queue_count',return_value=39),patch.object(main,'prepare') as prep:
            with self.assertRaises(ValueError):main.submit(self.root,self.p/'o')
            prep.assert_not_called()
    def test_job_budget_submission(self):
        suite=self.p/'fake';suite.mkdir();(suite/'logs').mkdir()
        def runner(args,**kw):return types.SimpleNamespace(returncode=0,stdout='1234\n',stderr='')
        with patch.object(main,'queue_count',return_value=35),patch.object(main,'prepare',return_value=suite),patch.object(main.subprocess,'run',side_effect=runner) as run:
            main.submit(self.root,self.p/'o')
            calls=[c.args[0] for c in run.call_args_list]
            self.assertTrue(any('--array=0-1' in c for c in calls));self.assertTrue(any('--dependency=afterany:1234' in c for c in calls))
    def test_submission_rejection_preserved(self):
        suite=self.p/'fake';suite.mkdir()
        with patch.object(main,'queue_count',return_value=0),patch.object(main,'prepare',return_value=suite),patch.object(main.subprocess,'run',return_value=types.SimpleNamespace(returncode=1,stdout='',stderr='LIMIT')):
            with self.assertRaises(RuntimeError):main.submit(self.root,self.p/'o')
        self.assertEqual(dat.read(suite/'submission.json')['submissions'][0]['status'],'REJECTED')
    def test_timeout_acceptance_unknown(self):
        suite=self.p/'fake';suite.mkdir()
        with patch.object(main,'queue_count',return_value=0),patch.object(main,'prepare',return_value=suite),patch.object(main.subprocess,'run',side_effect=subprocess.TimeoutExpired('sbatch',60)):
            with self.assertRaises(subprocess.TimeoutExpired):main.submit(self.root,self.p/'o')
        self.assertEqual(dat.read(suite/'submission.json')['submissions'][0]['status'],'ACCEPTANCE_UNKNOWN')

if __name__=='__main__':unittest.main()
