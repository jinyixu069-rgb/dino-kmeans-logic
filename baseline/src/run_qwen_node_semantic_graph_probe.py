"""Attach centered-mask Qwen embeddings to every SAM2 graph node."""
from __future__ import annotations
import csv,json,os
from pathlib import Path
import numpy as np, torch
from PIL import Image,ImageDraw,ImageFont
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from code.logic_prism.qwen3_vl_embedding_adapter import Qwen3VLEmbedder
from .common import metric_set,upper_tail_evidence
from .run_sam2_component_graph_probe import DATA, FLAVORS, graph_features

ROOT=Path(__file__).resolve().parents[2]
MODEL_PATH=Path(os.environ.get('QWEN3_VL_EMBEDDING_PATH',ROOT/'models/Qwen3-VL-Embedding-8B'))
TRACKED=ROOT/'baseline/results/sam2_component_graph_probe/tracked'
GRAPH_ROOT=ROOT/'baseline/results/sam2_component_graph_probe'
OUTPUT=ROOT/'baseline/results/qwen_node_semantic_graph_probe'
INSTRUCTION='Represent the isolated visual component for comparison of its appearance and semantic identity. Ignore the neutral canvas.'
SPLIT_DIR={'train_good':'train/good','test_good':'test/good','test_logical':'test/logical_anomalies','test_structural':'test/structural_anomalies'}

def font(size):
 p=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf');return ImageFont.truetype(str(p),size) if p.exists() else ImageFont.load_default()

def canvas_for_mask(image,mask,size=512,occupancy=.80,fill=(127,127,127)):
 mask=np.asarray(mask,bool);canvas=Image.new('RGB',(size,size),fill)
 yy,xx=np.where(mask)
 if not len(yy):return canvas
 box=(int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1));rgb=image.crop(box);alpha=Image.fromarray(mask[box[1]:box[3],box[0]:box[2]].astype(np.uint8)*255)
 scale=min(size*occupancy/rgb.width,size*occupancy/rgb.height);shape=(max(1,int(round(rgb.width*scale))),max(1,int(round(rgb.height*scale))))
 rgb=rgb.resize(shape,Image.Resampling.LANCZOS);alpha=alpha.resize(shape,Image.Resampling.LANCZOS);position=((size-shape[0])//2,(size-shape[1])//2)
 canvas.paste(rgb,position,alpha);return canvas

def item_canvas_set(path,masks):
 image=Image.open(path).convert('RGB');masks=np.asarray(masks,bool);background=~np.any(masks,axis=0)
 return [canvas_for_mask(image,m) for m in list(masks)+[background]]

def load_items():
 selection=json.loads((GRAPH_ROOT/'train_selection.json').read_text());items=[]
 for flavor,names in selection.items():
  for name in names:items.append({'split':'train_good','name':name,'flavor':flavor,'path':DATA/'train/good'/f'{name}.png'})
 for row in csv.DictReader(open(GRAPH_ROOT/'scores.csv')):
  items.append({'split':row['split'],'name':row['name'],'flavor':row['flavor'],'path':DATA/SPLIT_DIR[row['split']]/f"{row['name']}.png",'old_geometry':float(row['mahalanobis'])})
 for item in items:
  z=np.load(TRACKED/item['flavor']/item['split']/f"{item['name']}.npz");item['masks']=z['masks'].astype(bool);item['confidence']=z['confidence'];item['graph_feature']=graph_features(item['masks'],item['confidence'])
 return items

def cache_embeddings(items,batch_size=8):
 pending=[x for x in items if not (OUTPUT/'embeddings'/x['flavor']/x['split']/f"{x['name']}.npy").exists()]
 if not pending:return
 print(f'Loading Qwen; {len(pending)} images x 8 nodes',flush=True)
 model=Qwen3VLEmbedder(model_name_or_path=str(MODEL_PATH),output_dim=512,min_pixels=4096,max_pixels=262144,device='cuda',torch_dtype=torch.bfloat16)
 for index,item in enumerate(pending):
  canvases=item_canvas_set(item['path'],item['masks']);vectors=[]
  for start in range(0,len(canvases),batch_size):
   vectors.append(model.process([{'image':im,'instruction':INSTRUCTION} for im in canvases[start:start+batch_size]],normalize=True).float().cpu().numpy())
  out=OUTPUT/'embeddings'/item['flavor']/item['split']/f"{item['name']}.npy";out.parent.mkdir(parents=True,exist_ok=True);np.save(out,np.concatenate(vectors))
  if (index+1)%20==0:print(f'embedded {index+1}/{len(pending)} images',flush=True)
 del model;torch.cuda.empty_cache()

def semantic_scores(query,bank):
 # query [node,dim], bank [normal,node,dim]
 distances=1-np.einsum('id,nid->ni',query,bank)
 node_nearest=distances.min(axis=0);joint=np.sort(distances.mean(axis=1))[:3].mean()
 return node_nearest,float(joint)

def fit_geometry(items):
 x=np.stack([i['graph_feature'] for i in items]);scaler=StandardScaler().fit(x);z=scaler.transform(x);cov=LedoitWolf().fit(z)
 def score(item):
  q=scaler.transform(item['graph_feature'][None])[0];d=q-cov.location_;return float(d@cov.get_precision()@d)
 return score

def render_examples(items,path,title,with_scores=False):
 tile=150;head=28;rows=len(items);canvas=Image.new('RGB',(tile*9,38+rows*(tile+head)),'white');draw=ImageDraw.Draw(canvas);draw.text((8,7),title,fill='black',font=font(19))
 for r,item in enumerate(items):
  original=Image.open(item['path']).convert('RGB');original.thumbnail((tile,tile),Image.Resampling.LANCZOS);base=Image.new('RGB',(tile,tile),(127,127,127));base.paste(original,((tile-original.width)//2,(tile-original.height)//2));panels=[base]+item_canvas_set(item['path'],item['masks'])
  labels=['original']+[f'id{i}' for i in range(7)]+['background']
  for c,(panel,label) in enumerate(zip(panels,labels)):
   y=38+r*(tile+head);canvas.paste(panel.resize((tile,tile),Image.Resampling.LANCZOS),(c*tile,y+head));suffix=f" d={item['node_distances'][c-1]:.3f}" if with_scores and c else '';draw.text((c*tile+4,y+4),label+suffix,fill='black',font=font(12))
 canvas.save(path)

def main():
 OUTPUT.mkdir(parents=True,exist_ok=True);items=load_items();cache_embeddings(items)
 for item in items:item['embedding']=np.load(OUTPUT/'embeddings'/item['flavor']/item['split']/f"{item['name']}.npy").astype(np.float32)
 references={};models={};calibration={}
 for flavor in FLAVORS:
  train=[x for x in items if x['split']=='train_good' and x['flavor']==flavor];fit,cal=train[:20],train[20:]
  bank=np.stack([x['embedding'] for x in fit]);geo=fit_geometry(fit);models[flavor]=(bank,geo);calibration[flavor]={'semantic':[],'geometry':[]}
  for item in cal:
   nd,joint=semantic_scores(item['embedding'],bank);calibration[flavor]['semantic'].append(joint);calibration[flavor]['geometry'].append(geo(item))
 test=[x for x in items if x['split']!='train_good']
 for item in test:
  bank,geo=models[item['flavor']];nd,joint=semantic_scores(item['embedding'],bank);item['node_distances']=nd;item['semantic_joint']=joint;item['semantic_max_node']=float(nd.max());item['geometry_20']=geo(item)
  cal=calibration[item['flavor']];item['semantic_evidence']=float(upper_tail_evidence(np.asarray(cal['semantic']),np.asarray([joint]))[0]);item['geometry_evidence']=float(upper_tail_evidence(np.asarray(cal['geometry']),np.asarray([item['geometry_20']]))[0]);item['fused']=item['semantic_evidence']+item['geometry_evidence']
 subtypes=np.asarray([{'test_good':'good','test_logical':'logical_anomalies','test_structural':'structural_anomalies'}[x['split']] for x in test]);metrics={}
 for scorer in ('semantic_joint','semantic_max_node','geometry_20','fused'):metrics[scorer]=metric_set(subtypes,np.asarray([x[scorer] for x in test]))
 fields=['split','name','flavor','semantic_joint','semantic_max_node','geometry_20','semantic_evidence','geometry_evidence','fused']+[f'node_{i}_distance' for i in range(8)]
 with (OUTPUT/'scores.csv').open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=fields);w.writeheader()
  for x in test:w.writerow({'split':x['split'],'name':x['name'],'flavor':x['flavor'],**{k:x[k] for k in fields[3:9]},**{f'node_{i}_distance':float(v) for i,v in enumerate(x['node_distances'])}})
 anchors=[next(x for x in items if x['split']=='train_good' and x['name']==name) for name in ('000','002','004')];render_examples(anchors,OUTPUT/'centered_component_canvases.png','Centered mask-ID component canvases')
 logical=[x for x in test if x['split']=='test_logical'];rescued=sorted(logical,key=lambda x:(x['semantic_joint'],-x['geometry_20']),reverse=True)[:6];render_examples(rescued,OUTPUT/'semantic_anomaly_nodes.png','Logical anomalies with high node-semantic distance',True)
 (OUTPUT/'summary.json').write_text(json.dumps({'nodes':'7 tracked SAM2 IDs + 1 derived background','embedding_dim':512,'fit_train_per_flavor':20,'calibration_train_per_flavor':10,'metrics':metrics},indent=2)+'\n');print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__':main()
