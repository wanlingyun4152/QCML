#!/usr/bin/env python3
"""Predict parameters from input-only test data with a user-supplied joint checkpoint.

No target parameters or reference energies are loaded. Prediction files are
written only when this script is explicitly run with an external checkpoint.
"""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/joint_n2/engine'))
from check_test_inputs import load_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--inputs',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--threads',type=int,default=1)
    args = parser.parse_args()
    if args.threads<1:
        parser.error('--threads must be positive')
    out = args.out.expanduser().resolve()
    test_root = (ROOT/'tests').resolve()
    if out.exists() or out==test_root or test_root in out.parents:
        parser.error('Choose a new output directory outside tests/')
    data = load_inputs(args.inputs)
    import torch
    from joint import SharedTransformer
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if checkpoint.get('mode')!='joint':
        parser.error('Supply the joint model checkpoint')
    model = SharedTransformer()
    model.load_state_dict(checkpoint['state'],strict=True)
    model.eval()
    x = torch.as_tensor(data['features'],dtype=torch.float32)
    with torch.no_grad():
        pred = torch.cat([model(batch,str(data['domain'].item())) for batch in x.split(64)])
    values = pred.cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite model prediction')
    out.mkdir(parents=True,exist_ok=False)
    ids = {key:data[key] for key in ('index','global_indices') if key in data}
    np.savez_compressed(out/'predicted_parameters.npz',theta=values,**ids)
    print('PREDICTION_WRITTEN '+str(out/'predicted_parameters.npz'))


if __name__=='__main__':
    main()
