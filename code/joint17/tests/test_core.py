import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from qrev.common import save_weights,read_weights,sha256,outside,load_json
from qrev.models import FeatureTransformer,MLP,make_model
from qrev.data import NUMERIC,FEATURES,SYSTEMS,fit_scaler,encode_features,split_groups
from qrev.metrics import masked_loss,parameter_metrics,batch_invariance,predict
from qrev.train import epoch_order
from qrev.evaluate import evaluate
from qrev.collect import collect

ROOT=Path(__file__).resolve().parent.parent

def raw_frame(n=30):
    rng=np.random.RandomState(44)
    d=pd.DataFrame(rng.normal(size=(n,8)),columns=NUMERIC)
    d['system']=[SYSTEMS[i%len(SYSTEMS)] for i in range(n)]
    d['FCI_energy']=-10.0
    d['VQE_energy']=-9.9
    d['VQE_param1']=rng.normal(size=n)
    return d

def schema_for(d):
    return {'numeric_features':NUMERIC,'systems':SYSTEMS,'scaler':fit_scaler(d[NUMERIC].to_numpy(),np.arange(15))}

class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)
        self.x=np.random.RandomState(4).normal(size=(11,14)).astype(np.float32)
        self.small=FeatureTransformer(14,7,d_model=16,heads=2,layers=2,ff_dim=32,dropout=0.2)

    def test_01_input_has_no_label_or_energy(self):
        self.assertEqual(len(FEATURES),14)
        self.assertFalse(any('energy' in c.lower() or 'VQE_param' in c for c in FEATURES))

    def test_02_train_only_scaler(self):
        x=np.arange(160,dtype=float).reshape(20,8)
        a=fit_scaler(x,np.arange(10));x[10:]+=1e6
        self.assertEqual(a,fit_scaler(x,np.arange(10)))

    def test_03_constant_scaler(self):
        s=fit_scaler(np.ones((20,8)),np.arange(10))
        self.assertEqual(s['scale'],[1.0]*8)

    def test_04_encode_no_mutation(self):
        d=raw_frame();original=d.copy(deep=True);s=schema_for(d)
        a=encode_features(d,s);b=encode_features(d,s)
        pd.testing.assert_frame_equal(d,original,check_exact=True)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.shape,(30,14))

    def test_05_unknown_system_refused(self):
        d=raw_frame();s=schema_for(d);d.loc[0,'system']='H6'
        with self.assertRaises(ValueError): encode_features(d,s)

    def test_06_group_split_no_cross_ansatz_overlap(self):
        rows=[{'system':s,'geometry_group':s+'|'+str(i),'task':a+'/'+s}
              for s in ('H2','LiH') for i in range(30) for a in ('ucc','uccg')]
        d=pd.DataFrame(rows); labels,indices=split_groups(d)
        d['split']=labels
        self.assertTrue((d.groupby('geometry_group').split.nunique()==1).all())
        self.assertEqual(sorted(np.concatenate(list(indices.values())).tolist()),list(range(len(d))))

    def test_07_split_deterministic(self):
        d=pd.DataFrame([{'system':'H2','geometry_group':'H2|'+str(i),'task':'ucc/H2'} for i in range(100)])
        a,_=split_groups(d,42);b,_=split_groups(d,42);c,_=split_groups(d,43)
        np.testing.assert_array_equal(a,b);self.assertFalse(np.array_equal(a,c))

    def test_08_weighted_definition(self):
        p=torch.tensor([[1.,3.],[2.,8.]],requires_grad=True)
        y=torch.tensor([[0.,0.],[0.,float('nan')]])
        m=~torch.isnan(y)
        self.assertEqual(masked_loss(p,y,m,'weighted').item(),7.0)
        self.assertEqual(masked_loss(p,y,m,'unweighted').item(),4.5)

    def test_09_nan_mask_has_finite_gradient(self):
        p=torch.tensor([[1.,2.]],requires_grad=True)
        y=torch.tensor([[0.,float('nan')]]);m=~torch.isnan(y)
        masked_loss(p,y,m).backward()
        np.testing.assert_array_equal(p.grad.numpy(),[[2.,0.]])

    def test_10_zero_valid_label_is_not_deleted(self):
        p=torch.tensor([[2.]],requires_grad=True);y=torch.zeros_like(p)
        masked_loss(p,y,torch.ones_like(p,dtype=torch.bool)).backward()
        self.assertEqual(p.grad.item(),4.)

    def test_11_empty_mask_refused(self):
        p=torch.zeros(1,3)
        with self.assertRaises(ValueError):masked_loss(p,p,torch.zeros_like(p,dtype=torch.bool))

    def test_12_tokens_are_within_sample(self):
        z=self.small.tokens(torch.from_numpy(self.x))
        self.assertEqual(tuple(z.shape),(11,15,16))
        self.assertTrue(all(l.self_attn.batch_first for l in self.small.layers))

    def test_13_transformer_invariance(self):
        r=batch_invariance(self.small,self.x,torch.device('cpu'))
        self.assertTrue(r['passed'],r)

    def test_14_mlp_invariance(self):
        m=MLP(14,7,hidden=(16,16),dropout=0.2)
        self.assertTrue(batch_invariance(m,self.x,torch.device('cpu'))['passed'])

    def test_15_invariance_detects_bad_context_model(self):
        class Bad(nn.Module):
            def forward(self,x):return x+x.mean(0,keepdim=True)
        self.assertFalse(batch_invariance(Bad(),self.x,torch.device('cpu'))['passed'])

    def test_16_checkpoint_reload(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'best.pt';save_weights(p,self.small)
            a,_=predict(self.small,self.x,torch.device('cpu'))
            second=copy.deepcopy(self.small)
            second.load_state_dict(read_weights(p),strict=True)
            b,_=predict(second,self.x,torch.device('cpu'))
            np.testing.assert_array_equal(a,b)

    def test_17_best_last_separate(self):
        with tempfile.TemporaryDirectory() as td:
            p,q=Path(td)/'best.pt',Path(td)/'last.pt'
            h=save_weights(p,self.small)
            with torch.no_grad():self.small.head.bias.add_(1.)
            save_weights(q,self.small)
            self.assertEqual(sha256(p),h)
            self.assertNotEqual(sha256(q),h)

    def test_18_dropout_mode_restore(self):
        self.small.train();predict(self.small,self.x,torch.device('cpu'))
        self.assertFalse(self.small.training)
        self.small.train()
        self.assertTrue(all(m.training for m in self.small.modules()))
        self.assertTrue(all(m.training for m in self.small.modules() if isinstance(m,nn.Dropout)))

    def test_19_invariance_preserves_callers_mode(self):
        self.small.train();batch_invariance(self.small,self.x,torch.device('cpu'))
        self.assertTrue(self.small.training)

    def test_20_layer_initializations_independent(self):
        self.assertFalse(torch.equal(self.small.layers[0].linear1.weight,self.small.layers[1].linear1.weight))

    def test_21_batch_order_independent_model_rng(self):
        a=epoch_order(30,0,3);torch.rand(1000);np.random.rand(2000)
        np.testing.assert_array_equal(a,epoch_order(30,0,3))
        self.assertFalse(np.array_equal(a,epoch_order(30,0,4)))

    def test_22_metric_definitions(self):
        p=np.array([[1.,3.],[2.,9.]])
        y=np.array([[0.,0.],[0.,np.nan]]);m=np.isfinite(y)
        r=parameter_metrics(p,y,m)
        self.assertEqual(r['n_valid_parameters'],3)
        self.assertAlmostEqual(r['sample_mean_SSE'],7.)
        self.assertAlmostEqual(r['sample_mean_MSE'],4.5)
        self.assertAlmostEqual(r['parameter_RMSE_micro'],np.sqrt(14/3))

    def test_23_paths_are_read_only_protected(self):
        with tempfile.TemporaryDirectory() as d:
            for p in [d,Path(d)/'nested']:
                with self.assertRaises(ValueError):outside(p,d)
            outside(Path(d).parent/'unrelated',d)

    def test_24_test_evaluation_requires_confirmation(self):
        with self.assertRaisesRegex(ValueError,'locked'):
            evaluate('unused','unused',split='test')

    def test_25_model_capacities_close(self):
        nums=[]
        for name in ('transformer','mlp'):
            c=load_json(ROOT/'configs'/(name+'.json'))
            nums.append(sum(p.numel() for p in make_model(c,14,303).parameters()))
        self.assertLess(abs(nums[0]-nums[1])/max(nums),0.01)

    def test_26_collector_omits_weights_and_arrays(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d)/'suite';r.mkdir()
            (r/'best.pt').write_bytes(b'not copied')
            (r/'data.npz').write_bytes(b'not copied')
            (r/'status.json').write_text('{"status":"STARTED"}')
            result=collect(r,Path(d))
            import tarfile
            with tarfile.open(result['archive']) as t:
                names=t.getnames()
            self.assertFalse(any(n.endswith(('.pt','.npz')) for n in names))
            self.assertLess(result['bytes'],12*1024*1024)

    def test_27_loss_unweighted_scale_is_not_silently_normalized(self):
        p=torch.ones(2,5);y=torch.zeros_like(p);m=torch.ones_like(p,dtype=torch.bool)
        self.assertEqual(masked_loss(p,y,m,'weighted'),5*masked_loss(p,y,m,'unweighted'))

    def test_28_changing_energy_columns_does_not_change_input(self):
        d=raw_frame();s=schema_for(d);a=encode_features(d,s)
        d['FCI_energy']=np.arange(len(d))*1e8;d['VQE_param1']=1e6
        np.testing.assert_array_equal(a,encode_features(d,s))

    def test_29_collector_global_budget(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'suite';root.mkdir()
            for i in range(3):
                sub=root/str(i);sub.mkdir();(sub/'metrics.json').write_text('x'*600)
            with patch('qrev.collect.CONTENT_BUDGET',700):
                r=collect(root,Path(d))
            manifest=load_json(Path(r['archive'][:-7])/'collection_manifest.json')
            self.assertLessEqual(manifest['copied_payload_bytes'],700)
            self.assertTrue(any(f['status']=='omitted_global_budget' for f in manifest['files']))

    def test_30_collector_hard_archive_limit(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'suite';root.mkdir()
            with patch('qrev.collect.ARCHIVE_HARD_LIMIT',10):
                with self.assertRaises(RuntimeError):collect(root,Path(d))
            self.assertFalse(list(Path(d).glob('*.tar.gz')))

    def test_31_nan_features_refused(self):
        d=raw_frame();s=schema_for(d);d.loc[0,'bondlengths']=float('nan')
        with self.assertRaises(ValueError):encode_features(d,s)

    def test_32_shape_errors_refused(self):
        with self.assertRaises(ValueError):self.small(torch.zeros(2,1,14))
        with self.assertRaises(ValueError):MLP(14,3)(torch.zeros(2,15))

if __name__=='__main__':unittest.main()
