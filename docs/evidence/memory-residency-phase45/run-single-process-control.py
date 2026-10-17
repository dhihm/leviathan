import datetime,hashlib,json,re,statistics,subprocess,time
from pathlib import Path
b=Path(__file__).parent;tooling=b.parent;p=json.loads((b/'protocol.json').read_text());old=json.loads((tooling/'phase43-ir-discovery-shadow/next-baseline-noise-protocol.json').read_text());exe=Path(old['baselineExecutable']);plan=Path(old['plan']);composite=Path(old['composite']);assert hashlib.sha256(exe.read_bytes()).hexdigest()==old['baselineSha256'];files=[exe,plan,composite];before={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in files}
lock=tooling/'moe-root-gpu-lock.json';assert json.loads(lock.read_text()).get('gpuLock')!='running'
prior=json.loads((plan.parent/'serving-three-results.json').read_text());expected=next(x['generatedIDs'] for x in prior['fixtures'] if x['fixture']=='raw-hello')
args=[str(exe),'gemma-generate','--file','<workspace>/models/gemma-4-26B-A4B-Q4_0/gemma-4-26B_q4_0-it.gguf','--graph',str(tooling/'gemma-graph.json'),'--plan',str(plan),'--composite-plan',str(composite),'--moe-region',str(tooling/'moe-region-root.json'),'--text','Hello','--max-tokens','32','--repeat','32','--bos-align']
def observe(label):
 raw={}
 for name,cmd in [('memory',['vm_stat']),('thermal',['pmset','-g','therm']),('power',['pmset','-g','batt']),('load',['uptime'])]:
  x=subprocess.run(cmd,capture_output=True,text=True,timeout=10);raw[name]={'exit':x.returncode,'stdout':x.stdout,'stderr':x.stderr}
 counters={}
 for key in ['Pageouts','Swapins','Swapouts']:
  m=re.search(r'^'+key+r':\s*(\d+)',raw['memory']['stdout'],re.M);counters[key]=int(m.group(1)) if m else None
 return {'label':label,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'raw':raw,'memoryCounters':counters}
out=b/'single-process-control';out.mkdir(exist_ok=False);r={'scope':'Single-process frozen-baseline residency reproducibility control, not candidate comparison. Prior failed A/A remains failed. Global observations do not prove quiet or causality.','protocol':p['singleProcessControl'],'command':args,'startedUtc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'before':observe('before-control')};lock.write_text(json.dumps({'gpuLock':'running','mode':'phase45-single-process-residency-control'}))
try:
 t=time.monotonic()
 with (out/'run.log').open('x') as log:proc=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=300)
 r['elapsedSeconds']=time.monotonic()-t;r['exit']=proc.returncode;r['after']=observe('after-control');txt=(out/'run.log').read_text();r['decodeTotalsMs']=[float(x) for x in re.findall(r'repetition \d+/32:.*?decode: ([0-9.]+) ms total \(31 steps\)',txt)];r['generatedIDs']=re.findall(r'^generated ids: (.+)$',txt,re.M);r['fallbackValues']=[int(x) for x in re.findall(r'fallback (\d+)',txt)];r['metalAllocatedGiB']=re.findall(r'metal device allocated sample: ([0-9.]+)',txt)
 assert proc.returncode==0 and 'GENERATION_COMPLETED' in txt and len(r['decodeTotalsMs'])==32 and len(r['generatedIDs'])==32
 assert all(x==expected for x in r['generatedIDs']) and len(r['fallbackValues'])>=9 and all(x==0 for x in r['fallbackValues'])
 r['blocks']=[]
 for i in range(7):
  totals=r['decodeTotalsMs'][4+i*4:8+i*4];r['blocks'].append({'block':i+1,'repetitionIndices':[5+i*4,6+i*4,7+i*4,8+i*4],'decodeTotalsMs':totals,'warmMedianMsPerToken':statistics.median(totals[1:])/31})
 med=[x['warmMedianMsPerToken'] for x in r['blocks']];r['overallMedianMsPerToken']=statistics.median(med);r['relativeSpread']=(max(med)-min(med))/r['overallMedianMsPerToken'];r['memoryActivityDeltas']={k:r['after']['memoryCounters'][k]-r['before']['memoryCounters'][k] for k in r['before']['memoryCounters']};r['allBlocksRetained']=True;r['filesUnchanged']=before=={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in files};assert r['filesUnchanged'];r['allGeneratedIDsMatchValidatedBaseline']=True;r['allReportedFallbackZero']=True;r['spreadPass']=r['relativeSpread']<=0.05;r['pagingPass']=r['memoryActivityDeltas']['Pageouts']==0 and r['memoryActivityDeltas']['Swapouts']==0;r['completedUtc']=datetime.datetime.now(datetime.timezone.utc).isoformat();r['performanceComparisonPermitted']=r['spreadPass'] and r['pagingPass'];(out/'results.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:r[k] for k in ['exit','elapsedSeconds','blocks','relativeSpread','memoryActivityDeltas','performanceComparisonPermitted']},indent=2),flush=True)
except BaseException as err:
 r['failure']=repr(err);(out/'failed.json').write_text(json.dumps(r,indent=2));raise
finally:lock.write_text(json.dumps({'gpuLock':'finished','mode':'phase45-single-process-residency-control'}))
