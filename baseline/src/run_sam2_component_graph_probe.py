"""Pure SAM2 mask-ID graph baseline for juice_bottle logical anomalies."""
from __future__ import annotations
import csv,json,os,tempfile
from pathlib import Path
import numpy as np, torch
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from .common import metric_set

ROOT=Path(__file__).resolve().parents[2]
DATA=Path('/mnt/nfs/xujy/logicdataset/dataset_loco/juice_bottle')
QWEN=ROOT/'baseline/results/qwen_juice_flavor_grouping'
AMG=ROOT/'baseline/results/sam2_amg_probe'
SAM2_ROOT=ROOT/'third_party/sam2'; CHECKPOINT=ROOT/'models/sam2.1_hiera_large.pt'
OUTPUT=ROOT/'baseline/results/sam2_component_graph_probe'
FLAVORS=['orange','banana','cherry'];ANCHORS={'orange':'000','banana':'002','cherry':'004'}
SPLITS={'test_good':'test/good','test_logical':'test/logical_anomalies','test_structural':'test/structural_anomalies'}

def palette(n):
 base=[(245,80,80),(70,160,255),(80,210,120),(250,185,55),(190,75,220),(55,215,215),(245,125,65),(150,220,90)]
 return [base[i%len(base)] for i in range(n)]

def norm(x):
 x=np.asarray(x,np.float32);return x/max(float(np.linalg.norm(x)),1e-12)

def prepare_routing(train_per_flavor=30):
 rows=list(csv.DictReader(open(QWEN/'assignments_dim512.csv')));by={f:[] for f in FLAVORS}
 for r in rows:
  name=r['basename'];vec=norm(np.load(QWEN/'embeddings_4096'/f'{name}.npy')[:512]);by[r['predicted_flavor']].append((name,vec))
 centroids={f:norm(np.mean([v for _,v in values],axis=0)) for f,values in by.items()}
 selected={}
 for f,values in by.items():
  values=sorted(values,key=lambda nv:1-float(nv[1]@centroids[f]));names=[n for n,_ in values[:train_per_flavor]]
  if ANCHORS[f] not in names:names[-1]=ANCHORS[f]
  selected[f]=names
 routed=[]
 for split,subdir in SPLITS.items():
  cache=QWEN/'embeddings_4096'/split
  for path in sorted((DATA/subdir).glob('*.png')):
   vec=norm(np.load(cache/f'{path.stem}.npy')[:512]);similar={f:float(vec@c) for f,c in centroids.items()};flavor=max(similar,key=similar.get)
   routed.append({'split':split,'name':path.stem,'path':path,'flavor':flavor,'similarities':similar})
 return selected,routed

def track(predictor,source_image,target_image,source_masks):
 with tempfile.TemporaryDirectory(dir='/dev/shm') as td:
  d=Path(td);source_image.save(d/'000.jpg',quality=95);target_image.resize(source_image.size,Image.Resampling.LANCZOS).save(d/'001.jpg',quality=95)
  state=predictor.init_state(video_path=str(d),offload_video_to_cpu=True)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   for i,m in enumerate(source_masks,1):predictor.add_new_mask(state,frame_idx=0,obj_id=i,mask=m)
   result=None
   for frame,ids,logits in predictor.propagate_in_video(state,start_frame_idx=0,max_frame_num_to_track=1,reverse=False):
    if frame==1:
     order={int(v):j for j,v in enumerate(ids)};raw=np.stack([logits[order[i],0].float().cpu().numpy() for i in range(1,len(source_masks)+1)])
     result=raw>0
  predictor.reset_state(state)
 if result is None:raise RuntimeError('no frame-1 masks')
 confidence=np.asarray([float((1/(1+np.exp(-np.clip(r,-20,20))))[m].mean()) if m.any() else 0 for r,m in zip(raw,result)])
 return result,confidence

def graph_features(masks,confidence):
 masks=np.asarray(masks,bool);h,w=masks.shape[1:];nodes=[]
 for m,q in zip(masks,confidence):
  yy,xx=np.where(m);area=float(m.mean())
  if len(yy):
   cx=float((xx.mean()+.5)/w);cy=float((yy.mean()+.5)/h);bw=float((xx.max()-xx.min()+1)/w);bh=float((yy.max()-yy.min()+1)/h)
   perimeter=float((m^ndimage.binary_erosion(m)).sum()/max(np.sqrt(m.sum()),1))
  else:cx=cy=bw=bh=perimeter=0.
  nodes.extend([area,cx,cy,bw,bh,perimeter,float(q)])
 edges=[]
 for a in range(len(masks)):
  for b in range(a+1,len(masks)):
   ma,mb=masks[a],masks[b];aa=max(int(ma.sum()),1);ab=max(int(mb.sum()),1);inter=int((ma&mb).sum());union=max(int((ma|mb).sum()),1)
   ya,xa=np.where(ma);yb,xb=np.where(mb);cxa=float(xa.mean()/w) if len(xa) else 0;cya=float(ya.mean()/h) if len(ya) else 0;cxb=float(xb.mean()/w) if len(xb) else 0;cyb=float(yb.mean()/h) if len(yb) else 0
   edges.extend([cxb-cxa,cyb-cya,float(np.hypot(cxb-cxa,cyb-cya)),float(np.log(ab/aa)),inter/union,inter/aa,inter/ab])
 return np.asarray(nodes+edges,np.float64)

def font(size):
 p=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf');return ImageFont.truetype(str(p),size) if p.exists() else ImageFont.load_default()

def render_sheet(items,path,title):
 pw,ph=240,480;cols=4;rows=int(np.ceil(len(items)/cols));canvas=Image.new('RGB',(pw*cols,40+ph*rows),'white');draw=ImageDraw.Draw(canvas);draw.text((8,8),title,fill='black',font=font(20))
 colors=palette(7)
 for k,item in enumerate(items):
  image=Image.open(item['path']).convert('RGB');values=np.asarray(image).astype(np.float32)
  for i,m in enumerate(item['masks']):values[m]=values[m]*.78+np.asarray(colors[i],np.float32)*.22
  panel=Image.fromarray(values.astype(np.uint8));panel.thumbnail((pw,ph-28),Image.Resampling.LANCZOS);cell=Image.new('RGB',(pw,ph),'white');cell.paste(panel,((pw-panel.width)//2,28));ImageDraw.Draw(cell).text((5,4),f"{item['split']} {item['name']} {item['flavor']} s={item['score']:.1f}",fill='black',font=font(13));canvas.paste(cell,((k%cols)*pw,40+(k//cols)*ph))
 canvas.save(path)

def main():
 OUTPUT.mkdir(parents=True,exist_ok=True);cache=OUTPUT/'tracked';cache.mkdir(exist_ok=True)
 selected,routed=prepare_routing();(OUTPUT/'train_selection.json').write_text(json.dumps(selected,indent=2)+'\n')
 os.chdir(SAM2_ROOT);from sam2.build_sam import build_sam2_video_predictor
 print('Loading SAM2 video predictor',flush=True);predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',str(CHECKPOINT),device='cuda')
 all_items=[]
 jobs=[]
 for flavor,names in selected.items():
  for name in names:jobs.append(('train_good',name,DATA/'train/good'/f'{name}.png',flavor,None))
 for r in routed:jobs.append((r['split'],r['name'],r['path'],r['flavor'],r['similarities']))
 anchors={f:Image.open(DATA/'train/good'/f'{a}.png').convert('RGB') for f,a in ANCHORS.items()};source_masks={f:np.load(AMG/a/'amg_masks.npz')['masks'].astype(bool) for f,a in ANCHORS.items()}
 for index,(split,name,path,flavor,similarities) in enumerate(jobs):
  out=cache/flavor/split/f'{name}.npz';out.parent.mkdir(parents=True,exist_ok=True)
  if out.exists():z=np.load(out);masks=z['masks'].astype(bool);confidence=z['confidence']
  else:
   masks,confidence=track(predictor,anchors[flavor],Image.open(path).convert('RGB'),source_masks[flavor]);np.savez_compressed(out,masks=masks,confidence=confidence)
  all_items.append({'split':split,'name':name,'path':path,'flavor':flavor,'similarities':similarities,'masks':masks,'confidence':confidence,'feature':graph_features(masks,confidence)})
  if (index+1)%20==0:print(f'tracked {index+1}/{len(jobs)}',flush=True)
 models={}
 for flavor in FLAVORS:
  train=np.stack([x['feature'] for x in all_items if x['split']=='train_good' and x['flavor']==flavor]);scaler=StandardScaler().fit(train);z=scaler.transform(train);models[flavor]=(scaler,LedoitWolf().fit(z))
 for item in all_items:
  scaler,cov=models[item['flavor']];z=scaler.transform(item['feature'][None])[0];d=z-cov.location_;item['mahalanobis']=float(d@cov.get_precision()@d);item['topk_z']=float(np.mean(np.sort(z*z)[-10:]));item['score']=item['mahalanobis']
 test=[x for x in all_items if x['split']!='train_good'];subtypes=np.asarray([{'test_good':'good','test_logical':'logical_anomalies','test_structural':'structural_anomalies'}[x['split']] for x in test])
 for item in test:item['qwen_distance']=float(1.0-max(item['similarities'].values()))
 summary={}
 for scorer in ('mahalanobis','topk_z','qwen_distance'):
  summary[scorer]=metric_set(subtypes,np.asarray([x[scorer] for x in test]))
 fields=['split','name','flavor','qwen_orange','qwen_banana','qwen_cherry','qwen_distance','mahalanobis','topk_z','mean_track_confidence']
 with (OUTPUT/'scores.csv').open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=fields);w.writeheader()
  for x in test:w.writerow({'split':x['split'],'name':x['name'],'flavor':x['flavor'],**{f'qwen_{f}':x['similarities'][f] for f in FLAVORS},'qwen_distance':x['qwen_distance'],'mahalanobis':x['mahalanobis'],'topk_z':x['topk_z'],'mean_track_confidence':float(np.mean(x['confidence']))})
 logical=sorted([x for x in test if x['split']=='test_logical'],key=lambda x:x['score'],reverse=True);good=sorted([x for x in test if x['split']=='test_good'],key=lambda x:x['score'],reverse=True)
 render_sheet(logical[:16],OUTPUT/'top_logical.png','Highest-scoring logical anomalies');render_sheet(logical[-16:],OUTPUT/'missed_logical.png','Lowest-scoring logical anomalies');render_sheet(good[:16],OUTPUT/'false_positive_good.png','Highest-scoring normal images')
 (OUTPUT/'summary.json').write_text(json.dumps({'train_per_flavor':{f:len(v) for f,v in selected.items()},'n_test':len(test),'metrics':summary,'known_limitation':'tracked mask IDs encode geometry but do not verify component semantics or discover extra components'},indent=2)+'\n')
 print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
