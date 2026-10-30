#!/usr/bin/env python3
"""Integrity tests on actual renders and a trusted locally generated model."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
from sorter import Sorter, LABELS, COLORS, p

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--artifacts',default='outputs');ap.add_argument('--output',default='integrity_tests.json');a=ap.parse_args()
 root=Path(a.artifacts);model=joblib.load(root/'color_mlp.joblib')
 data=np.load(root/'dataset'/'features.npz',allow_pickle=False)
 metrics=json.loads((root/'evaluation.json').read_text())
 manifest=json.loads((root/'dataset'/'manifest.json').read_text())
 tests=[]
 test=data['split']=='test';pred=model.predict(data['X'][test]);correct=int(np.sum(pred==data['y'][test]))
 tests.append({'name':'saved_test_metric_recomputed','passed':correct==metrics['test_correct'] and int(test.sum())==metrics['test_total'],'correct':correct,'total':int(test.sum())})
 train_files={r['file'] for r in manifest if r['split']=='train'}
 test_files={r['file'] for r in manifest if r['split']=='test'}
 tests.append({'name':'render_split_filenames_disjoint','passed':not bool(train_files&test_files) and len(train_files)==135 and len(test_files)==45,'train':len(train_files),'test':len(test_files)})
 s=Sorter(Path('.'),False)
 try:
  for _ in range(120):p.stepSimulation(physicsClientId=s.client)
  refused=[]
  for lab in LABELS:
   cid,detail=s.attach_virtual_grasp(s.objects[lab]);refused.append(cid is None and not detail['contact_verified'])
   if cid is not None:p.removeConstraint(cid,physicsClientId=s.client)
  tests.append({'name':'no_contact_no_virtual_grasp','passed':all(refused),'refused_count':sum(refused),'total':3})
  # Same bodies and metadata, changed visible RGB colors: predictions must follow appearance.
  appearance={'RED':'BLUE','GREEN':'RED','BLUE':'GREEN'}
  for body_label,color_label in appearance.items():
   p.changeVisualShape(s.objects[body_label],-1,rgbaColor=COLORS[color_label],physicsClientId=s.client)
  observed={lab:s.classify_object(model,lab)[0] for lab in LABELS}
  tests.append({'name':'prediction_follows_rendered_rgb_not_body_identity','passed':observed==appearance,'expected':appearance,'observed':observed})
 finally:s.close()
 result={'all_passed':all(t['passed'] for t in tests),'tests':tests}
 Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
 return 0 if result['all_passed'] else 1
if __name__=='__main__':raise SystemExit(main())
