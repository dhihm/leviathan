import subprocess,json,time,datetime,re,select
from pathlib import Path
b=Path(__file__).parent
model='<workspace>/models/gemma-4-26B-A4B-Q4_0/gemma-4-26B_q4_0-it.gguf'
def observe(label):
 raw={}
 for name,cmd in [('memory',['vm_stat']),('thermal',['pmset','-g','therm']),('power',['pmset','-g','batt']),('load',['uptime'])]:
  r=subprocess.run(cmd,capture_output=True,text=True,timeout=10);raw[name]={'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
 mem=raw['memory']['stdout'];counts={}
 for key in ['Pageouts','Swapins','Swapouts','Pages wired down','Pages occupied by compressor','Pages free','Pages active','Pages inactive','File-backed pages']:
  m=re.search(r'^'+re.escape(key)+r':\s*(\d+)',mem,re.M);counts[key]=int(m.group(1)) if m else None
 return {'label':label,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'raw':raw,'memoryCounters':counts}
r={'scope':'Own staged diagnosis, global telemetry cannot attribute all changes to the child','startedUtc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phases':[],'before':observe('before-probe')}
p=subprocess.Popen([str(b/'memory-probe'),model],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
try:
 while True:
  if not select.select([p.stdout],[],[],45)[0]:raise TimeoutError('No own probe phase within45 seconds')
  line=p.stdout.readline()
  if not line:break
  if line.strip()=='LOCAL_MEMORY_PROBE_COMPLETED':r['completedMarker']=True;break
  stage=json.loads(line);stage['beforeHold']=observe(stage['phase']+'-before-hold');time.sleep(15);stage['afterHold']=observe(stage['phase']+'-after-hold');r['phases'].append(stage)
  (b/'probe-progress.json').write_text(json.dumps(r,indent=2));print(json.dumps({'phase':stage['phase'],'memoryBefore':stage['beforeHold']['memoryCounters'],'memoryAfter':stage['afterHold']['memoryCounters']}),flush=True)
  p.stdin.write('continue\n');p.stdin.flush()
 r['exit']=p.wait(timeout=15);r['stderr']=p.stderr.read();r['after']=observe('after-probe');r['completedUtc']=datetime.datetime.now(datetime.timezone.utc).isoformat();assert r['exit']==0 and r.get('completedMarker');(b/'probe-results.json').write_text(json.dumps(r,indent=2))
except BaseException as e:
 r['failure']=repr(e);(b/'probe-failed.json').write_text(json.dumps(r,indent=2));raise
finally:
 if p.poll() is None:p.terminate();p.wait(timeout=10)
