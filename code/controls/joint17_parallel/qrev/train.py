"""Validation-only model selection; test label file is never opened here."""
import csv
import hashlib
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from .common import (VERSION,dump_json,load_json,sha256,obj_hash,new_dir,setup,
                     environment,save_weights,read_weights,source_snapshot,sync)
from .data import load_prepared
from .models import make_model
from .metrics import masked_loss,predict,parameter_metrics,batch_invariance


def epoch_order(n,seed,epoch):
    # Independent of model initialization/dropout RNG usage.
    return np.random.RandomState((seed+104729*epoch+173)%2**32).permutation(n)


def validate_config(c):
    t = c['training']
    for field in ('epochs','batch_size','scheduler_patience'):
        if int(t[field]) < 1:
            raise ValueError(field+' must be positive')
    if not 0<=int(c['seed'])<2**31 or t['lr']<=0 or t['weight_decay']<0 or t['max_grad_norm']<=0:
        raise ValueError('Invalid optimizer/seed configuration')
    if t['loss'] not in ('weighted','unweighted') or not 0<t['scheduler_factor']<1:
        raise ValueError('Invalid loss/scheduler')
    if c.get('selection_metric') != 'sample_mean_MSE':
        raise ValueError('This controlled v1 uses one fixed validation selection metric')


def train(prepared,config_path,out,seed=None,epochs=None,loss=None,device='cpu',threads=1,purpose='pilot'):
    begin = time.perf_counter()
    config = load_json(config_path)
    if seed is not None: config['seed']=int(seed)
    if epochs is not None: config['training']['epochs']=int(epochs)
    if loss is not None: config['training']['loss']=loss
    config['purpose']=purpose
    validate_config(config)
    prepared = Path(prepared).expanduser().resolve()
    manifest,schema,meta,a = load_prepared(prepared,('train','val'))
    out = new_dir(out,[manifest['data_root'],Path(manifest['data_root']).parent/'train',prepared])
    status = {'version':VERSION,'status':'STARTED','purpose':purpose,'test_labels_opened':False,
              'out':str(out),'prepared':str(prepared),'prepared_manifest_sha256':sha256(prepared/'manifest.json'),
              'config_sha256':obj_hash(config),'source_fingerprint':manifest['source_fingerprint']}
    dump_json(out/'status.json',status)
    try:
        dev = setup(config['seed'],device,threads)
        dump_json(out/'environment.json',{**environment(),'device':str(dev),'threads':threads,
                   'deterministic_algorithms':True,'tf32':False})
        dump_json(out/'config.json',config)
        dump_json(out/'inference_schema.json',schema)
        status['inference_schema_sha256'] = sha256(out/'inference_schema.json')
        status['source_code_fingerprint'] = source_snapshot(out)
        model = make_model(config,schema['input_dim'],schema['output_dim']).to(dev)
        status['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        X,Y,M = a['train']['X'],a['train']['Y'],a['train']['mask']
        # Strictly training/validation labels. No test.npz load in this module.
        x = torch.as_tensor(X,dtype=torch.float32)
        y = torch.as_tensor(np.where(M,Y,0.0),dtype=torch.float32)
        m = torch.as_tensor(M,dtype=torch.bool)
        cfg = config['training']
        opt = torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',
             factor=cfg['scheduler_factor'],patience=cfg['scheduler_patience'],min_lr=cfg['min_lr'])
        best_value,best_epoch = float('inf'),None
        best_sha = None
        order_hash = hashlib.sha256()
        mode_failures = 0
        history_columns = ['epoch','train_objective','validation_sample_mean_MSE',
                           'validation_parameter_MAE_micro','validation_parameter_RMSE_micro',
                           'lr_used','lr_next','epoch_seconds','training_mode_restored','epoch_order_sha256']
        history_path = out/'history.tsv'
        sync(dev)
        training_start = time.perf_counter()
        with history_path.open('w',encoding='utf8',newline='') as h:
            writer = csv.DictWriter(h,fieldnames=history_columns,delimiter='\t'); writer.writeheader()
            for epoch in range(1,cfg['epochs']+1):
                epstart = time.perf_counter()
                model.train()  # Must be restored after EACH validation call.
                train_mode_ok = all(module.training for module in model.modules())
                mode_failures += int(not train_mode_ok)
                if not train_mode_ok: raise AssertionError('Training mode not restored')
                order = epoch_order(len(X),config['seed'],epoch)
                order_bytes = order.astype('<i8').tobytes()
                order_hash.update(order_bytes)
                epoch_sha = hashlib.sha256(order_bytes).hexdigest()
                train_sum = 0.0
                lr_used = float(opt.param_groups[0]['lr'])
                for start in range(0,len(order),cfg['batch_size']):
                    idx = torch.as_tensor(order[start:start+cfg['batch_size']],dtype=torch.long)
                    xb,yb,mb = x[idx].to(dev),y[idx].to(dev),m[idx].to(dev)
                    opt.zero_grad(set_to_none=True)
                    pred = model(xb)
                    value = masked_loss(pred,yb,mb,cfg['loss'])
                    if not torch.isfinite(value): raise FloatingPointError('Nonfinite train loss')
                    value.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['max_grad_norm'],error_if_nonfinite=True)
                    opt.step()
                    train_sum += float(value.detach().cpu())*len(idx)
                pv,_ = predict(model,a['val']['X'],dev,cfg['batch_size'])
                vm = parameter_metrics(pv,a['val']['Y'],a['val']['mask'])
                val_metric = vm['sample_mean_MSE']
                scheduler.step(val_metric)  # Same selection metric for both loss variants.
                if val_metric < best_value:
                    best_value,best_epoch = val_metric,epoch
                    best_sha = save_weights(out/'best.pt',model)
                    dump_json(out/'best_meta.json',{'epoch':epoch,'selection_split':'val',
                        'selection_metric':'sample_mean_MSE','value':best_value,
                        'weights_sha256':best_sha,'config_sha256':obj_hash(config),
                        'prepared_manifest_sha256':status['prepared_manifest_sha256']})
                sync(dev)
                row = {'epoch':epoch,'train_objective':train_sum/len(X),
                       'validation_sample_mean_MSE':val_metric,
                       'validation_parameter_MAE_micro':vm['parameter_MAE_micro'],
                       'validation_parameter_RMSE_micro':vm['parameter_RMSE_micro'],
                       'lr_used':lr_used,'lr_next':float(opt.param_groups[0]['lr']),
                       'epoch_seconds':time.perf_counter()-epstart,
                       'training_mode_restored':train_mode_ok,'epoch_order_sha256':epoch_sha}
                writer.writerow(row); h.flush()
                if epoch==1 or epoch==cfg['epochs'] or epoch%cfg['log_every']==0:
                    print('%s seed=%d epoch=%d train=%.8g val_sample_mse=%.8g best_epoch=%d' %
                         (config['model'],config['seed'],epoch,row['train_objective'],val_metric,best_epoch),flush=True)
        sync(dev)
        status['training_loop_seconds_including_validation_and_best_saves'] = time.perf_counter()-training_start
        last_sha = save_weights(out/'last.pt',model)
        dump_json(out/'last_meta.json',{'epoch':cfg['epochs'],'weights_sha256':last_sha,
                                      'not_used_for_model_selection':True})
        if sha256(out/'best.pt') != best_sha: raise AssertionError('Best checkpoint was overwritten')
        # Verify strict checkpoint reload before the actual invariance acceptance.
        best = make_model(config,schema['input_dim'],schema['output_dim']).to(dev)
        best.load_state_dict(read_weights(out/'best.pt'),strict=True)
        pv,_ = predict(best,a['val']['X'],dev,cfg['batch_size'])
        metrics = parameter_metrics(pv,a['val']['Y'],a['val']['mask'])
        if not np.isclose(metrics['sample_mean_MSE'],best_value,rtol=1e-6,atol=1e-10):
            raise AssertionError('Best checkpoint reload failed to reproduce selected validation score')
        check = batch_invariance(best,a['val']['X'],dev,contexts=a['train']['X'],
                                **config['invariance'])
        dump_json(out/'batch_invariance.json',check)
        dump_json(out/'validation_metrics.json',metrics)
        status.update({'best_epoch':best_epoch,'epochs_completed':cfg['epochs'],
            'best_validation_sample_mean_MSE':best_value,'best_weights_sha256':best_sha,
            'last_weights_sha256':last_sha,'batch_order_sha256':order_hash.hexdigest(),
            'training_mode_failures':mode_failures,'checkpoint_reload_passed':True,
            'invariance_passed':check['passed'],'test_labels_opened':False,
            'input_dim':schema['input_dim'],'output_dim':schema['output_dim']})
        if not check['passed']:
            raise AssertionError('Batch invariance failed at the declared tolerance; do not use as independent model')
        # Verify read-only use of prepared train/val/schema, not test labels.
        load_prepared(prepared,('train','val'))
        status['status']='COMPLETED'
    except Exception as exc:
        status.update({'status':'FAILED','error_type':type(exc).__name__,'error':str(exc),
                       'traceback':traceback.format_exc(limit=12)})
        raise
    finally:
        status['total_command_seconds'] = time.perf_counter()-begin
        dump_json(out/'status.json',status)
    print('TRAIN_STATUS = COMPLETED\nrun_dir = '+str(out),flush=True)
    return status
