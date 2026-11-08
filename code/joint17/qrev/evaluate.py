"""Explicit evaluation/inference and metadata-safe export."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .common import (dump_json,load_json,obj_hash,sha256,new_dir,setup,read_weights,write_tsv)
from .data import load_prepared,encode_features
from .models import make_model
from .metrics import predict,parameter_metrics,sample_metric_rows,term_metric_rows


def load_run(run,device='cpu',threads=1):
    run = Path(run).expanduser().resolve()
    status = load_json(run/'status.json')
    if status.get('status') != 'COMPLETED' or not status.get('invariance_passed'):
        raise ValueError('Run is incomplete or failed invariance acceptance')
    cfg = load_json(run/'config.json')
    if obj_hash(cfg) != status['config_sha256']:
        raise ValueError('Run configuration changed after training')
    schema = load_json(run/'inference_schema.json')
    if sha256(run/'inference_schema.json') != status['inference_schema_sha256']:
        raise ValueError('Inference schema changed after training')
    best = load_json(run/'best_meta.json')
    if sha256(run/'best.pt') != best['weights_sha256'] or best['weights_sha256'] != status['best_weights_sha256']:
        raise ValueError('Best checkpoint hash mismatch')
    dev = setup(cfg['seed'],device,threads)
    model = make_model(cfg,schema['input_dim'],schema['output_dim']).to(dev)
    model.load_state_dict(read_weights(run/'best.pt'),strict=True)
    model.eval()
    return model,schema,cfg,status,dev


def evaluate(run,prepared,split='val',device='cpu',threads=1,confirm_final_test=False):
    if split=='test' and not confirm_final_test:
        raise ValueError('Final test is locked. Freeze comparison configurations first, then use --confirm-final-test')
    if split not in ('train','val','test'):
        raise ValueError('Unknown split')
    start = time.perf_counter()
    run,prepared = Path(run).resolve(),Path(prepared).resolve()
    model,schema,cfg,status,dev = load_run(run,device,threads)
    if sha256(prepared/'manifest.json') != status['prepared_manifest_sha256']:
        raise ValueError('Prepared split differs from the training split')
    manifest,schema2,meta,arrays = load_prepared(prepared,(split,))
    if obj_hash(schema) != obj_hash(schema2):
        raise ValueError('Prepared inference schema mismatch')
    a = arrays[split]
    target_meta = meta.iloc[a['global_indices']].copy(deep=True).reset_index(drop=True)
    before = target_meta.copy(deep=True)
    out = new_dir(run/('evaluation_'+split),[manifest['data_root'],prepared])
    pred,inference_seconds = predict(model,a['X'],dev,cfg['training']['batch_size'])
    metrics = parameter_metrics(pred,a['Y'],a['mask'])
    metrics.update({'split':split,'best_epoch':status['best_epoch'],
                    'model':cfg['model'],'seed':cfg['seed'],'loss':cfg['training']['loss'],
                    'purpose':cfg['purpose'],'reference':'Stored VQE parameters, without periodic wrapping',
                    'energy_evaluated':False,'quantum_circuit_evaluated':False,
                    'prediction_seconds_including_tensor_transfer_and_output_concatenation':inference_seconds,
                    'final_test_explicitly_unlocked':bool(split=='test')})
    # Preserve metadata in original units. Never inverse-transform it, never mutate prepared metadata.
    target_meta.to_csv(out/'prediction_metadata.tsv',sep='\t',index=False,float_format='%.17g')
    np.savez_compressed(out/'predictions.npz',prediction=pred,
                        target=a['Y'],mask=a['mask'],global_indices=a['global_indices'],
                        sample_id=target_meta.sample_id.to_numpy(dtype=str),
                        parameter_columns=np.asarray(schema['parameter_columns'],dtype=str))
    write_tsv(out/'sample_metrics.tsv',sample_metric_rows(pred,a['Y'],a['mask'],target_meta))
    write_tsv(out/'term_metrics.tsv',term_metric_rows(pred,a['Y'],a['mask'],target_meta,schema['parameter_columns']))
    task_rows = []
    for task in sorted(target_meta.task.unique()):
        take = target_meta.task.to_numpy()==task
        task_rows.append({'task':task,**parameter_metrics(pred[take],a['Y'][take],a['mask'][take])})
    write_tsv(out/'task_metrics.tsv',task_rows)
    pd.testing.assert_frame_equal(before,target_meta,check_exact=True)
    metrics['metadata_unchanged']=True
    metrics['total_evaluation_seconds']=time.perf_counter()-start
    dump_json(out/'metrics.json',metrics)
    print('EVALUATION_COMPLETED split=%s out=%s' % (split,out),flush=True)
    return metrics


def predict_table(run,input_table,out,device='cpu',threads=1,batch_size=64):
    """Deployment-style inference, requiring only system + eight raw numeric features.

    Exports all 303 learned slots; it does NOT manufacture a complete UCC vector
    from missing columns or assume a general new-basis mapping.
    """
    p = Path(input_table).expanduser().resolve()
    model,schema,cfg,status,dev = load_run(run,device,threads)
    raw = pd.read_csv(p,sep='\t')
    snapshot = raw.copy(deep=True)
    x = encode_features(raw,schema)
    pred,seconds = predict(model,x,dev,batch_size)
    out = new_dir(out,[p.parent,Path(run).resolve()])
    exported = pd.concat([raw.reset_index(drop=True).copy(deep=True),
              pd.DataFrame(pred,columns=['pre_'+c for c in schema['parameter_columns']])],axis=1)
    exported.to_csv(out/'predictions.tsv.gz',sep='\t',index=False,compression='gzip',float_format='%.17g')
    pd.testing.assert_frame_equal(raw,snapshot,check_exact=True)
    dump_json(out/'prediction_report.json',{'run':str(Path(run).resolve()),'input_sha256':sha256(p),
        'rows':len(raw),'batch_size':batch_size,'seconds_including_transfer':seconds,
        'parameters':schema['parameter_columns'],'metadata_unchanged':True,
        'energy_evaluated':False,'missing_circuit_parameters_reconstructed':False})
    print('PREDICTION_COMPLETED',out,flush=True)
