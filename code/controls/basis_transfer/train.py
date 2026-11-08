"""Deterministic, validation-selected training. Never reads a test array."""
from pathlib import Path
import csv, hashlib, json, time, platform
import numpy as np
import torch
from models import initialize, features, predict, invariance, state_hash
from data import sha,check

def write(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def save_weights(p,state):
    tmp=p.with_name(p.name+'.tmp')
    torch.save({k:v.detach().cpu().clone() for k,v in state.items()},tmp);tmp.replace(p)

def load_weights(p):return torch.load(p,map_location='cpu',weights_only=True)

def metrics(pred,y):
    d=np.asarray(pred,dtype=np.float64)-np.asarray(y,dtype=np.float64)
    return {'MSE':float(np.mean(d*d)),'RMSE':float(np.sqrt(np.mean(d*d))),'MAE':float(np.mean(np.abs(d))),
            'samples':len(d),'parameters_per_sample':d.shape[1]}

def order_for(n,seed,epoch):
    return np.random.RandomState((int(seed)*100003+int(epoch)*1009+7205)%(2**32-1)).permutation(n)

def train_run(out,family,seed,train,val,epochs=3000,source=None,extra=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    cfg=dict(family=family,seed=int(seed),epochs=int(epochs),output_dim=train['theta'].shape[1],
       lr=.001,weight_decay=.001,batch_size=64,gradient_clip=.1,loss='mean squared parameter error; no output scaling',
       best_selection='validation parameter MSE',source_checkpoint=str(source) if source else None,
       source_checkpoint_sha256=sha(source) if source else None,
       source_backbone_only=source is not None,optimizer_state_transferred=False,
       selected_train_ids=train['sample_id'].tolist(),val_ids=val['sample_id'].tolist(),
       final_test_evaluated=False,**(extra or {}))
    write(out/'config.json',cfg);write(out/'status.json',{'status':'RUNNING','ML_final_test_evaluated':False})
    t0=time.perf_counter()
    try:
        state=load_weights(source) if source else None
        model=initialize(family,cfg['output_dim'],seed,state)
        head_hash=state_hash(model.head.state_dict());bb_hash=state_hash(model.backbone.state_dict())
        if state is not None:
            expected={k[9:]:v for k,v in state.items() if k.startswith('backbone.')}
            check(state_hash(expected)==bb_hash,'Transferred backbone mismatch')
        x=features(train['X_raw']);xv=features(val['X_raw']);y=train['theta'].astype(np.float32)
        tx=torch.from_numpy(x);ty=torch.from_numpy(y)
        torch.manual_seed(28000+int(seed))  # dropout stream independent of construction/transfer
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
        sched=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=50,min_lr=1e-6)
        best=float('inf');best_epoch=None;order_digest=hashlib.sha256();history=[]
        initial=metrics(predict(model,xv),val['theta'])
        write(out/'initialization.json',{'head_sha256':head_hash,'backbone_sha256':bb_hash,
                'initial_validation_metrics':initial,'parameters':sum(p.numel() for p in model.parameters()),
                'source_head_loaded':False,'source_backbone_loaded':source is not None})
        with (out/'history.tsv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['epoch','train_MSE','val_MSE','learning_rate','epoch_seconds'],delimiter='\t');w.writeheader()
            for epoch in range(1,epochs+1):
                st=time.perf_counter();model.train()
                check(all(m.training for m in model.modules()),'train mode not restored')
                order=order_for(len(tx),seed,epoch);order_digest.update(order.astype('<i8').tobytes())
                total=0.
                for start in range(0,len(tx),64):
                    ii=order[start:start+64]
                    optimizer.zero_grad(set_to_none=True)
                    output=model(tx[ii]);loss=torch.mean((output-ty[ii])**2)
                    check(bool(torch.isfinite(loss)),'Nonfinite training loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True)
                    optimizer.step();total+=float(loss.detach())*len(ii)
                pred=predict(model,xv);mv=metrics(pred,val['theta'])
                check(np.isfinite(mv['MSE']),'Nonfinite validation loss')
                if mv['MSE']<best:
                    best=mv['MSE'];best_epoch=epoch;save_weights(out/'best.pt',model.state_dict())
                    write(out/'best_meta.json',{'epoch':epoch,'validation_MSE':best,'weights_sha256':sha(out/'best.pt')})
                lr=float(optimizer.param_groups[0]['lr']);sched.step(mv['MSE'])
                row={'epoch':epoch,'train_MSE':total/len(tx),'val_MSE':mv['MSE'],'learning_rate':lr,'epoch_seconds':time.perf_counter()-st}
                history.append(row);w.writerow(row);f.flush()
                if epoch==1 or epoch%200==0 or epoch==epochs:
                    print('%s epoch=%d/%d val_MSE=%.9g best=%.9g'%(out.name,epoch,epochs,mv['MSE'],best),flush=True)
        save_weights(out/'last.pt',model.state_dict())
        train_seconds=time.perf_counter()-t0
        model.load_state_dict(load_weights(out/'best.pt'),strict=True)
        pred=predict(model,xv);mv=metrics(pred,val['theta'])
        check(abs(mv['MSE']-best)<=1e-12+1e-5*abs(best),'Best checkpoint replay failed')
        inv=invariance(model,xv,x);check(inv['passed'],'Cross-sample invariance failed')
        np.savez_compressed(out/'validation_predictions.npz',prediction=pred,label=val['theta'],
                            sample_id=val['sample_id'],R_angstrom=val['R_angstrom'])
        with (out/'validation_samples.tsv').open('w',newline='') as f:
            w=csv.writer(f,delimiter='\t');w.writerow(['sample_id','R_angstrom','parameter_RMSE','parameter_MAE'])
            for i in range(len(pred)):
                d=pred[i].astype(float)-val['theta'][i]
                w.writerow([str(val['sample_id'][i]),float(val['R_angstrom'][i]),float(np.sqrt(np.mean(d*d))),float(np.mean(abs(d)))])
        # Archive a documented sample of histories, NEVER present it as every epoch.
        selected=[v for v in history if v['epoch']<=5 or v['epoch']>epochs-5 or v['epoch']%20==0 or v['epoch']==best_epoch]
        with (out/'history_sampled.tsv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(history[0]),delimiter='\t');w.writeheader();w.writerows(selected)
        result={'status':'COMPLETED_VALIDATION_ONLY','family':family,'seed':seed,
                'n_train':len(x),'epochs':epochs,'best_epoch':best_epoch,'metrics':mv,
                'parameters':sum(p.numel() for p in model.parameters()),'training_seconds':train_seconds,
                'invariance':inv,'checkpoint_reload_passed':True,'best_sha256':sha(out/'best.pt'),
                'last_sha256':sha(out/'last.pt'),'train_order_sha256':order_digest.hexdigest(),
                'initial_head_sha256':head_hash,'initial_backbone_sha256':bb_hash,
                'source_checkpoint_sha256':cfg['source_checkpoint_sha256'],
                'source_cost_not_included_in_target_training_seconds':source is not None,
                'history_sha256':sha(out/'history.tsv'),'history_sampling':'first5,last5,every20,best epoch; full history stays on server',
                'ML_final_test_evaluated':False,'energy_evaluated':False,'old_paper_checkpoint_used':False,
                'python':platform.python_version(),'numpy':np.__version__,'torch':str(torch.__version__)}
        write(out/'result.json',result);write(out/'status.json',{'status':result['status'],'ML_final_test_evaluated':False})
        return result
    except BaseException as e:
        import traceback
        write(out/'status.json',{'status':'FAILED','error':repr(e),'traceback':traceback.format_exc(),'ML_final_test_evaluated':False})
        raise
