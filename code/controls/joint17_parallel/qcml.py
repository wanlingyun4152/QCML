#!/usr/bin/env python3
"""QCML independent-sample revision entry. Python 3.9+, NumPy, pandas, PyTorch 2.0+.
No sklearn, scipy, torch-geometric, remote download or environment modification.
"""
import argparse
import importlib
import inspect
import json
import os
import sys
import traceback
from pathlib import Path

# Set before torch is imported. CPU is the default; explicit CUDA must exist.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def check_env():
    result={'python':sys.version,'executable':sys.executable,'sklearn_required':False,'imports':{}}
    ready = sys.version_info >= (3,9)
    for name in ('numpy','pandas','torch'):
        try:
            m=importlib.import_module(name)
            result['imports'][name]={'ok':True,'version':str(m.__version__)}
            if name=='torch':
                ok='weights_only' in inspect.signature(m.load).parameters
                result['safe_weights_only_load_available']=ok
                ready &= ok
                ready &= int(m.__version__.split('.')[0])>=2
        except Exception as exc:
            result['imports'][name]={'ok':False,'error':repr(exc)};ready=False
    result['ready']=bool(ready)
    print(json.dumps(result,indent=2))
    print('environment_status = '+('READY' if ready else 'NOT_READY'))
    return 0 if ready else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('check-env')
    a=sub.add_parser('prepare')
    a.add_argument('--data-root',required=True);a.add_argument('--out',required=True)
    a.add_argument('--seed',type=int,default=42)
    a.add_argument('--val-fraction',type=float,default=0.15);a.add_argument('--test-fraction',type=float,default=0.15)
    a.add_argument('--allow-source-change',action='store_true')
    a=sub.add_parser('train')
    a.add_argument('--prepared',required=True);a.add_argument('--config',required=True);a.add_argument('--out',required=True)
    a.add_argument('--seed',type=int);a.add_argument('--epochs',type=int)
    a.add_argument('--loss',choices=['weighted','unweighted'])
    a.add_argument('--purpose',choices=['smoke','pilot','formal'],default='pilot')
    a.add_argument('--device',choices=['cpu','cuda'],default='cpu');a.add_argument('--threads',type=int,default=1)
    a=sub.add_parser('evaluate')
    a.add_argument('--run',required=True);a.add_argument('--prepared',required=True)
    a.add_argument('--split',choices=['train','val','test'],default='val');a.add_argument('--confirm-final-test',action='store_true')
    a.add_argument('--device',choices=['cpu','cuda'],default='cpu');a.add_argument('--threads',type=int,default=1)
    a=sub.add_parser('predict')
    a.add_argument('--run',required=True);a.add_argument('--input-table',required=True);a.add_argument('--out',required=True)
    a.add_argument('--batch-size',type=int,default=64)
    a.add_argument('--device',choices=['cpu','cuda'],default='cpu');a.add_argument('--threads',type=int,default=1)
    a=sub.add_parser('collect')
    a.add_argument('--root',required=True);a.add_argument('--out-parent')
    a=sub.add_parser('test')
    a.add_argument('--report',help='Optional JSON result path; parent must already exist')
    args=p.parse_args()
    if args.command=='check-env':return check_env()
    if args.command=='prepare':
        from qrev.data import prepare
        raise ValueError('Use parallel.py init to prepare the controlled LiH restriction of the parent split')
    elif args.command=='train':
        from qrev.train import train
        train(args.prepared,args.config,args.out,args.seed,args.epochs,args.loss,args.device,args.threads,args.purpose)
    elif args.command=='evaluate':
        from qrev.evaluate import evaluate
        evaluate(args.run,args.prepared,args.split,args.device,args.threads,args.confirm_final_test)
    elif args.command=='predict':
        from qrev.evaluate import predict_table
        predict_table(args.run,args.input_table,args.out,args.device,args.threads,args.batch_size)
    elif args.command=='collect':
        from qrev.collect import collect
        collect(args.root,args.out_parent)
    elif args.command=='test':
        import unittest
        import time
        suite=unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent/'tests'))
        start=time.perf_counter()
        res=unittest.TextTestRunner(verbosity=2).run(suite)
        if args.report:
            from qrev.common import dump_json
            dump_json(args.report,{'tests_run':res.testsRun,'failures':len(res.failures),'errors':len(res.errors),
                                 'skipped':len(res.skipped),'passed':res.wasSuccessful(),
                                 'seconds':time.perf_counter()-start})
        return 0 if res.wasSuccessful() else 2
    return 0

if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception:
        print('STATUS = FAILED',file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)
