import argparse, csv, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from config import ModelConfig, DataConfig
from datasets import make_dataset, collate_fn
from models.detector import JointDetector


def args():
    p=argparse.ArgumentParser(description='Evaluate trained lane head on BDD100K lane-labeled validation images')
    p.add_argument('--ckpt', required=True, help='Full joint checkpoint with trained lane head/backbone')
    p.add_argument('--data-dir', default=None, help='BDD100K root; if omitted uses DataConfig.bdd_dir')
    p.add_argument('--out-dir', default='runs/bdd_lane_eval')
    p.add_argument('--split', choices=['val','validation'], default='val')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--max-images', type=int, default=0, help='0 evaluates full split')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--save-vis', type=int, default=20, help='Number of labeled examples to visualize')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def state_dict(ck):
    if isinstance(ck, dict):
        for k in ('model','model_state','state_dict','model_state_dict'):
            if k in ck and isinstance(ck[k], dict): return ck[k]
    return ck


def calc(tp,fp,fn):
    pr=tp/(tp+fp) if tp+fp else 0.0
    re=tp/(tp+fn) if tp+fn else 0.0
    f=2*pr*re/(pr+re) if pr+re else 0.0
    return {'TP':int(tp),'FP':int(fp),'FN':int(fn),'precision':float(pr),'recall':float(re),'F1':float(f)}


def main():
    a=args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mc=ModelConfig(); dc=DataConfig()
    if a.data_dir: dc.bdd_dir=a.data_dir
    ds=make_dataset('bdd100k',mc,dc,split=a.split)
    dl=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.workers,
                   pin_memory=device.type=='cuda',collate_fn=collate_fn)
    model=JointDetector(mc).to(device)
    ck=torch.load(a.ckpt,map_location=device,weights_only=False)
    missing,unexpected=model.load_state_dict(state_dict(ck),strict=False)
    if any('lane_head' in k for k in missing):
        raise RuntimeError('Checkpoint does not contain lane_head weights.')
    print(f'Checkpoint: {a.ckpt}\nDevice: {device}\nMissing keys: {len(missing)} | unexpected keys: {len(unexpected)}')
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    all_p=[]; all_t=[]; slot_counts=[[0,0,0] for _ in range(mc.num_lanes)]
    labeled=0; unlabeled=0; vis=0
    if a.save_vis:
        import cv2
    mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1)
    model.eval()
    with torch.no_grad():
        for bi,(images,targets) in enumerate(dl):
            if a.max_images and bi*a.batch_size>=a.max_images: break
            if a.max_images:
                remain=a.max_images-bi*a.batch_size
                images=images[:remain]
                for k,v in targets.items():
                    if torch.is_tensor(v): targets[k]=v[:remain]
            images=images.to(device,non_blocking=True)
            gt=targets['lane_targets'].to(device).long()
            has=targets['has_lane_labels'].to(device).bool()
            logits=model(images)['lane_logits']
            probs=torch.softmax(logits,dim=-1)
            fgprob,fgidx=probs[...,1:].max(dim=-1)
            pred=(probs.argmax(dim=-1)!=0)&(fgprob>=a.threshold)
            pred_bins=torch.where(pred,fgidx+1,torch.zeros_like(fgidx))
            labeled+=int(has.sum()); unlabeled+=int((~has).sum())
            if not has.any(): continue
            pp=pred[has].cpu().numpy(); tt=(gt[has]>0).cpu().numpy()
            all_p.append(pp); all_t.append(tt)
            for li in range(mc.num_lanes):
                slot_counts[li][0]+=int((pp[:,li]&tt[:,li]).sum())
                slot_counts[li][1]+=int((pp[:,li]&~tt[:,li]).sum())
                slot_counts[li][2]+=int((~pp[:,li]&tt[:,li]).sum())
            if a.save_vis and vis<a.save_vis:
                # visualizations for only labeled examples; green=prediction, white=GT
                ims=images[has]
                pb=pred_bins[has].cpu().numpy(); tb=gt[has].cpu().numpy()
                for j in range(ims.shape[0]):
                    if vis>=a.save_vis: break
                    arr=(ims[j:j+1]*std+mean).clamp(0,1)[0].permute(1,2,0).cpu().numpy()
                    arr=(arr*255).astype(np.uint8); arr=cv2.cvtColor(arr,cv2.COLOR_RGB2BGR)
                    H,W=arr.shape[:2]
                    colors=[(0,255,0),(0,200,255),(255,0,0),(255,0,255)]
                    for li in range(mc.num_lanes):
                        pred_pts=[]; gt_pts=[]
                        for r in range(mc.grid_h):
                            y=int((r+.5)/mc.grid_h*H)
                            x1=int((pb[j,li,r]-1)/max(mc.grid_w-1,1)*(W-1)) if pb[j,li,r]>0 else -1
                            x2=int((tb[j,li,r]-1)/max(mc.grid_w-1,1)*(W-1) if tb[j,li,r]>0 else -1)
                            if x1>=0: pred_pts.append((x1,y))
                            if x2>=0: gt_pts.append((x2,y))
                        if len(pred_pts)>=2: cv2.polylines(arr,[np.array(pred_pts,np.int32)],False,colors[li%4],2)
                        if len(gt_pts)>=2: cv2.polylines(arr,[np.array(gt_pts,np.int32)],False,(255,255,255),2)
                    cv2.imwrite(str(out/f'bdd_lane_compare_{vis:04d}.jpg'),arr); vis+=1
    if not all_p: raise RuntimeError('No BDD100K lane-labeled validation examples were found. Check annotation files/schema.')
    p=np.concatenate(all_p); t=np.concatenate(all_t)
    tp=int((p&t).sum()); fp=int((p&~t).sum()); fn=int((~p&t).sum()); union=int((p|t).sum())
    overall=calc(tp,fp,fn); overall['occupancy_IoU']=float(tp/union if union else 1.0)
    per_slot=[]
    for i,(tp_i,fp_i,fn_i) in enumerate(slot_counts):
        row={'lane_slot':i,**calc(tp_i,fp_i,fn_i)}; per_slot.append(row)
    result={'checkpoint':a.ckpt,'dataset':'BDD100K','split':a.split,'confidence_threshold':a.threshold,
      'images_seen':labeled+unlabeled,'images_with_lane_labels':labeled,'images_without_lane_labels':unlabeled,
      'labeled_images_evaluated':int(p.shape[0]),'overall_occupancy':overall,'per_lane_slot':per_slot,
      'interpretation_note':'Cross-dataset evaluation: lane head trained on OpenLane, evaluated on BDD100K. These are custom row-anchor occupancy metrics, not official BDD100K lane benchmark metrics. Per-slot metrics are affected by BDD lane ordering/slot assignment.'}
    (out/'bdd_lane_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with open(out/'bdd_lane_per_slot.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(per_slot[0].keys())); w.writeheader(); w.writerows(per_slot)
    print(f"Images seen: {labeled+unlabeled} | lane-labeled: {labeled} | skipped unlabeled: {unlabeled}")
    print(f"Labeled images evaluated: {p.shape[0]} | threshold={a.threshold}")
    print(f"Overall occupancy P/R/F1: {overall['precision']:.4f}/{overall['recall']:.4f}/{overall['F1']:.4f} | IoU={overall['occupancy_IoU']:.4f}")
    for r in per_slot: print(f"Slot {r['lane_slot']}: P={r['precision']:.4f} R={r['recall']:.4f} F1={r['F1']:.4f} TP={r['TP']} FP={r['FP']} FN={r['FN']}")
    if a.save_vis: print(f'Saved {vis} prediction-vs-GT overlays to {out.resolve()}')
    print(f'JSON: {(out/"bdd_lane_metrics.json").resolve()}')
    print(f'CSV: {(out/"bdd_lane_per_slot.csv").resolve()}')

if __name__=='__main__': main()
