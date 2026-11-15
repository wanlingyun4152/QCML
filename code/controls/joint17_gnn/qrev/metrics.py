"""Parameter-space metrics only. No energy/chemical-accuracy claims."""
import time
import numpy as np
import torch
from .common import sync

def masked_loss(pred,target,mask,mode='weighted'):
    if pred.shape != target.shape or target.shape != mask.shape:
        raise ValueError('Loss tensor shapes differ')
    count = mask.sum(dim=1)
    if (count==0).any():
        raise ValueError('Zero-valid-label sample')
    # Mask BEFORE subtraction/squaring; NaNs must not contaminate gradients.
    target_safe = torch.where(mask,target,torch.zeros_like(target))
    err = torch.where(mask,pred-target_safe,torch.zeros_like(pred))
    sse = err.square().sum(1)
    if mode == 'weighted':
        return sse.mean()  # n_i * (sum err^2 / n_i), observed valid count, NOT no_parameters
    if mode == 'unweighted':
        return (sse/count).mean()
    raise ValueError('loss must be weighted or unweighted')

def parameter_metrics(pred,target,mask):
    p,y,m = np.asarray(pred,dtype=np.float64),np.asarray(target,dtype=np.float64),np.asarray(mask,dtype=bool)
    if p.shape != y.shape or y.shape != m.shape or not np.isfinite(p).all() or not m.any(1).all():
        raise ValueError('Invalid prediction/label arrays')
    err = np.zeros_like(p)
    err[m] = p[m]-y[m]
    c = m.sum(1)
    sse = np.square(err).sum(1)
    sae = np.abs(err).sum(1)
    return {'n_samples':len(p),'n_valid_parameters':int(c.sum()),
            'parameter_MAE_micro':float(sae.sum()/c.sum()),
            'parameter_RMSE_micro':float(np.sqrt(sse.sum()/c.sum())),
            'sample_mean_parameter_RMSE':float(np.sqrt(sse/c).mean()),
            'sample_mean_MSE':float((sse/c).mean()),
            'sample_mean_SSE':float(sse.mean())}

@torch.no_grad()
def predict(model,X,device,batch_size=64):
    """X only; no targets/other-geometry summaries are accepted by forward."""
    if batch_size < 1 or len(X)==0:
        raise ValueError('Invalid inference batch size or empty input')
    model.eval()
    result = []
    sync(device)
    t = time.perf_counter()
    for start in range(0,len(X),batch_size):
        tensor = torch.as_tensor(np.asarray(X[start:start+batch_size]),dtype=torch.float32,device=device)
        result.append(model(tensor).detach().cpu().numpy())
    sync(device)
    result = np.concatenate(result,axis=0)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite predictions')
    return result, time.perf_counter()-t

@torch.no_grad()
def batch_invariance(model,X,device,contexts=None,atol=1e-5,rtol=1e-5,seed=31415):
    """All output columns checked, not just label-valid columns. Eval-mode only.

    Tests original order, chunks, singleton inputs, set permutation and inserting
    additional *training* geometries. Thresholds are declared before running.
    """
    if len(X)==0 or atol < 0 or rtol < 0:
        raise ValueError('Invalid invariance inputs')
    was_training = model.training
    try:
        whole,_ = predict(model,X,device,len(X))
        candidates = {}
        candidates['chunks64'],_ = predict(model,X,device,64)
        candidates['single'],_ = predict(model,X,device,1)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(X))
        yp,_ = predict(model,X[perm],device,len(X))
        candidates['permuted_same_set'] = yp[np.argsort(perm)]
        extra = X[:min(32,len(X))] if contexts is None else contexts[:min(32,len(contexts))]
        padded = np.concatenate([extra,X,extra],axis=0)
        yc,_ = predict(model,padded,device,len(padded))
        candidates['other_geometries_added'] = yc[len(extra):len(extra)+len(X)]
        records = []
        for name,yp in candidates.items():
            delta = np.abs(yp.astype(np.float64)-whole.astype(np.float64))
            bound = atol+rtol*np.abs(whole.astype(np.float64))
            records.append({'case':name,'passed':bool((delta<=bound).all()),
                            'max_abs_delta_all_outputs':float(delta.max()),
                            'violating_output_elements':int((delta>bound).sum())})
        return {'passed':all(r['passed'] for r in records),'atol':atol,'rtol':rtol,
                'n_geometries':len(X),'n_extra_contexts_each_side':len(extra),
                'scope':'eval-mode output invariance; not prediction accuracy or energy accuracy',
                'cases':records}
    finally:
        model.train(was_training)

def sample_metric_rows(pred,y,mask,meta):
    err = np.zeros_like(pred,dtype=np.float64)
    err[mask] = pred[mask]-y[mask]
    count = mask.sum(1)
    rows = []
    for i,(_,r) in enumerate(meta.iterrows()):
        rows.append({'sample_id':r.sample_id,'task':r.task,'system':r.system,
            'bondlengths':float(r.bondlengths),'global_row':int(r.global_row),
            'valid_parameters':int(count[i]),'parameter_MAE':float(np.abs(err[i]).sum()/count[i]),
            'parameter_RMSE':float(np.sqrt(np.square(err[i]).sum()/count[i]))})
    return rows

def term_metric_rows(pred,y,mask,meta,param_names):
    rows = []
    tasks = meta.task.to_numpy()
    for task in sorted(set(tasks)):
        take = tasks==task
        for j,col in enumerate(param_names):
            valid = mask[take,j]
            if not valid.any():
                continue
            delta = pred[take,j][valid].astype(float)-y[take,j][valid]
            rows.append({'task':task,'parameter_column':col,'parameter_suffix_index':int(col[9:]),
                'n':int(valid.sum()),'MAE':float(np.abs(delta).mean()),
                'RMSE':float(np.sqrt(np.square(delta).mean())),
                'bias':float(delta.mean()),'abs_error_p50':float(np.quantile(np.abs(delta),0.5)),
                'abs_error_p90':float(np.quantile(np.abs(delta),0.9)),
                'abs_error_max':float(np.abs(delta).max())})
    return rows
