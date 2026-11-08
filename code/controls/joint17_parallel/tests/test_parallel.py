import sys,unittest,tempfile,tarfile,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qrev.models import LiHGraphNetwork,FeatureTransformer,MLP,make_model
from qrev.metrics import masked_loss,batch_invariance
from qrev.common import load_json,VERSION
from qrev.parallel import fingerprint
import collect_interfaces as ci

torch.set_num_threads(1)
class GraphTests(unittest.TestCase):
 def setUp(self):torch.manual_seed(8);self.x=torch.randn(7,14)
 def test_shape(self):self.assertEqual(LiHGraphNetwork(14,159)(self.x).shape,(7,159))
 def test_no_cross_geometry(self):
  g=LiHGraphNetwork(14,159).eval();r=batch_invariance(g,self.x,torch.device('cpu'),contexts=torch.randn(5,14));self.assertTrue(r['passed'])
 def test_permutation(self):
  g=LiHGraphNetwork(14,159).eval()
  torch.testing.assert_close(g.graph_embedding(self.x),g.graph_embedding(self.x,torch.tensor([1,0])),atol=1e-6,rtol=1e-6)
 def test_graph_distance_gradient(self):
  g=LiHGraphNetwork(14,159).eval();x=self.x.clone().requires_grad_();h=g.graph_embedding(x)
  (h[:,0].sum()).backward();self.assertGreater(x.grad[:,0].abs().sum().item(),0.)
  self.assertEqual(x.grad[:,1:].abs().sum().item(),0.)
 def test_wrong_dim(self):
  with self.assertRaises(ValueError):LiHGraphNetwork(13,159)
 def test_zero_targets_and_nan_mask(self):
  p=torch.tensor([[1.,2.,3.]],requires_grad=True);y=torch.tensor([[0.,float('nan'),1.]])
  m=torch.tensor([[True,False,True]])
  self.assertEqual(masked_loss(p,y,m,'weighted').item(),5.)
  self.assertEqual(masked_loss(p,y,m,'unweighted').item(),2.5)
  masked_loss(p,y,m).backward();self.assertEqual(p.grad[0,1].item(),0.)
 def test_parameter_budget(self):
  root=Path(__file__).resolve().parents[1];counts=[]
  for m in ('transformer','mlp','lih_gnn'):
   model=make_model(load_json(root/'configs'/f'{m}.json'),14,159)
   counts.append(sum(p.numel() for p in model.parameters()))
  self.assertEqual(counts,[112287,112565,112339]);self.assertLess(max(counts)/min(counts),1.01)
 def test_graph_features_not_labels(self):
  g=LiHGraphNetwork(14,159).eval()
  with self.assertRaises(TypeError):g(self.x,torch.randn(7,159))
 def test_source_manifest_has_new_files(self):
  f=fingerprint();self.assertIn('qrev/lih.py',f);self.assertIn('qrev/parallel.py',f);self.assertIn('collect_interfaces.py',f)

class CollectorTests(unittest.TestCase):
 def test_bounded_readonly_symlink(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)/'trans';root.mkdir();source=root/'vqe.py';source.write_text('import pyscf\nmol = pyscf.M(atom="Li 0 0 0; H 0 0 1.5")\n')
   before=source.read_bytes();(root/'large.dat').write_bytes(b'Z'*100000)
   other=Path(tmp)/'else.py';other.write_text('import pyscf\nSECRET_EXTERNAL=1\n');(root/'sym.py').symlink_to(other)
   out=Path(tmp)/'out';ar=ci.collect([root],out)
   with tarfile.open(ar) as t:
    names=t.getnames();self.assertTrue(any(n.endswith('vqe.py') for n in names));self.assertFalse(any(n.endswith('sym.py') for n in names));self.assertFalse(any(n.endswith('large.dat') for n in names))
   self.assertEqual(source.read_bytes(),before);self.assertLess(ar.stat().st_size,ci.ARCHIVE_LIMIT)
 def test_output_protection(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   with self.assertRaises(ValueError):ci.safe_parent(root/'out',[root])
 def test_missing_source(self):
  with tempfile.TemporaryDirectory() as tmp:
   with self.assertRaises(ValueError):ci.collect([Path(tmp)/'none'],Path(tmp)/'out')

if __name__=='__main__':unittest.main()
