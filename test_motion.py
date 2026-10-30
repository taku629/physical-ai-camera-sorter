#!/usr/bin/env python3
"""Run actual physics using an existing trusted local model; never fabricate results."""
import argparse,json
from pathlib import Path
import joblib
from sorter import Sorter

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--model',required=True)
 p.add_argument('--output',default='motion_test')
 a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 s=Sorter(out,False)
 try:
  rows=s.perform_sort(joblib.load(a.model))
  result={'sort_results':rows,'all_sort_success':all(r['success'] for r in rows),'video_created':False}
  (out/'motion_test.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
  print(json.dumps(result,indent=2),flush=True)
  return 0 if result['all_sort_success'] else 1
 finally:s.close()
if __name__=='__main__':raise SystemExit(main())
