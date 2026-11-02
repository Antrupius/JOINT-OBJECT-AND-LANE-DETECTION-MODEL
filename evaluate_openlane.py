import argparse, csv, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from config import ModelConfig, DataConfig
from datasets import make_dataset, collate_fn
from models.detector import JointDetector

def parse_args():
    p=argparse.ArgumentParser(description="Evaluate trained OpenLane lane head")
    p.add_argument("--ckpt", required=True, help="Trained lane checkpoint")
    p.add_argument("--out-dir", default="runs/openlane_eval")
    p.add_argument("--split", choices=["train","val"], default="val")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max-batches", type=int, default=0, help="0=full split")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--save-vis", type=int, default=0, help="Number of overlay images")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def get_state(obj):
    if isinstance(obj, dict):
        return obj.get(
            "model",
            obj.get("model_state", obj.get("state_dict", obj))
        )
    return obj

def metrics(p, t):
    p, t = p.astype(bool), t.astype(bool)
    tp=int((p&t).sum()); fp=int((p&~t).sum()); fn=int((~p&t).sum())
    pr=tp/(tp+fp) if tp+fp else 0.0
    re=tp/(tp+fn) if tp+fn else 0.0
    f1=2*pr*re/(pr+re) if pr+re else 0.0
    union=(p|t).sum()
    iou=(p&t).sum()/union if union else 1.0
    return dict(TP=tp,FP=fp,FN=fn,precision=float(pr),recall=float(re),F1=float(f1),occupancy_IoU=float(iou))

def main():
    a=parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mc=ModelConfig(); dc=DataConfig()
    ds=make_dataset("openlane",mc,dc,split=a.split)
    dl=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.workers,
                   pin_memory=device.type=="cuda",collate_fn=collate_fn)
    model=JointDetector(mc).to(device)
    ck=torch.load(a.ckpt,map_location=device,weights_only=False)
    state=get_state(ck)
    missing,unexpected=model.load_state_dict(state,strict=False)
    if any("lane_head" in k for k in missing):
        raise RuntimeError("Checkpoint lacks lane_head weights; choose trained OpenLane checkpoint.")
    print(f"Loaded: {a.ckpt} | device={device} | missing keys={len(missing)} unexpected={len(unexpected)}")
    model.eval(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    pred_all=[]; target_all=[]; lane_counts=[]
    vis=0
    if a.save_vis:
        import cv2
    with torch.no_grad():
        for bi,batch in enumerate(dl):
            if a.max_batches and bi>=a.max_batches: break
            images,targets=batch
            images=images.to(device,non_blocking=True)
            lt=targets["lane_targets"].to(device,non_blocking=True).long()
            has=targets["has_lane_labels"].to(device).bool()
            logits=model(images)["lane_logits"]
            probs=torch.softmax(logits,dim=-1)
            fgprob,fgidx=probs[...,1:].max(dim=-1)
            pred=(probs.argmax(dim=-1)!=0)&(fgprob>=a.threshold)
            bins=torch.where(pred,fgidx+1,torch.zeros_like(fgidx))
            if not has.any(): continue
            pred=pred[has]; bins=bins[has]; lt=lt[has]; imgs=images[has]
            p=pred.cpu().numpy(); t=(lt>0).cpu().numpy()
            pred_all.append(p); target_all.append(t)
            for li in range(p.shape[1]):
                while len(lane_counts)<=li: lane_counts.append([0,0,0])
                lane_counts[li][0]+=int((p[:,li]&t[:,li]).sum())
                lane_counts[li][1]+=int((p[:,li]&~t[:,li]).sum())
                lane_counts[li][2]+=int((~p[:,li]&t[:,li]).sum())
            if a.save_vis and vis<a.save_vis:
                mean=np.array([.485,.456,.406],np.float32); std=np.array([.229,.224,.225],np.float32)
                bp=bins.cpu().numpy(); bt=lt.cpu().numpy()
                for j in range(imgs.shape[0]):
                    if vis>=a.save_vis: break
                    im=imgs[j].cpu().permute(1,2,0).numpy()
                    im=np.clip(im*std+mean,0,1); im=(im*255).astype(np.uint8)
                    im=cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
                    H,W=im.shape[:2]; anchors=bp.shape[2]
                    colors=[(0,255,0),(0,200,255),(255,0,0),(255,0,255)]
                    for li in range(bp.shape[1]):
                        pp=[]; gg=[]
                        for row in range(anchors):
                            y=int(row/max(anchors-1,1)*(H-1))
                            xbin=int(bp[j,li,row]); gbin=int(bt[j,li,row])
                            if xbin>0: pp.append((int((xbin-1)/max(mc.grid_w-1,1)*(W-1)),y))
                            if gbin>0: gg.append((int((gbin-1)/max(mc.grid_w-1,1)*(W-1)),y))
                        if len(pp)>=2: cv2.polylines(im,[np.array(pp,np.int32)],False,colors[li%4],2)
                        if len(gg)>=2: cv2.polylines(im,[np.array(gg,np.int32)],False,(255,255,255),2)
                    cv2.imwrite(str(out/f"lane_overlay_{vis:04d}.jpg"),im); vis+=1
    if not pred_all: raise RuntimeError("No labeled lane examples evaluated; verify OpenLane paths.")
    p=np.concatenate(pred_all); t=np.concatenate(target_all)
    overall=metrics(p,t)
    rows=[]
    for li,(tp,fp,fn) in enumerate(lane_counts):
        pr=tp/(tp+fp) if tp+fp else 0.; re=tp/(tp+fn) if tp+fn else 0.
        f=2*pr*re/(pr+re) if pr+re else 0.
        rows.append(dict(lane_slot=li,TP=tp,FP=fp,FN=fn,precision=pr,recall=re,F1=f))
    result=dict(checkpoint=str(a.ckpt),split=a.split,threshold=a.threshold,
       examples_evaluated=int(p.shape[0]),overall=overall,per_lane_slot=rows,
       note="Internal row-anchor occupancy metrics, NOT official OpenLane benchmark metrics.")
    (out/"lane_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    with open(out/"per_lane_slot.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"Examples: {p.shape[0]} | threshold={a.threshold}")
    print(f"Overall occupancy P/R/F1: {overall['precision']:.4f}/{overall['recall']:.4f}/{overall['F1']:.4f}")
    print(f"Occupancy IoU: {overall['occupancy_IoU']:.4f}")
    for r in rows: print(f"Lane {r['lane_slot']}: P={r['precision']:.4f} R={r['recall']:.4f} F1={r['F1']:.4f} TP={r['TP']} FP={r['FP']} FN={r['FN']}")
    if a.save_vis: print(f"Saved {vis} overlays in {out.resolve()}")
    print(f"JSON: {(out/'lane_metrics.json').resolve()}")
    print(f"CSV: {(out/'per_lane_slot.csv').resolve()}")

if __name__=="__main__": main()
