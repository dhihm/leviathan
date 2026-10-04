import hashlib,json,subprocess,time
from pathlib import Path
r=Path('/Users/dhihm/Documents/Codex/2026-10-02/task-5/leviathan-compile-diagnostics-phase41');e=Path('/tmp/leviathan-forge2-tooling/phase41-compiler-integrity-review/production-lifecycle');lock=e.parent.parent/'moe-root-gpu-lock.json';assert json.loads(lock.read_text())['gpuLock']=='finished'
exe=r/'.build/release/leviathan-cli';model=r.parent/'models/gemma-4-26B-A4B-Q4_0/gemma-4-26B_q4_0-it.gguf';graph=e.parent.parent/'gemma-graph.json';region=e.parent.parent/'moe-region-root.json';plan=e/'batch-plan.json';comp=e/'composite-pass.json';assert not plan.exists() and not comp.exists()
args=[str(exe),'gemma-forge','--file',str(model),'--graph',str(graph),'--plan',str(plan),'--text','Hello','--gen-tokens','32','--composite-plan',str(comp),'--moe-region',str(region)]
before={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'Sources').rglob('*.swift')};meta={'command':args,'binarySha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'modelBytes':model.stat().st_size,'modelDownloaded':False,'sourceBefore':before,'stages':[],'startedUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};lock.write_text(json.dumps({'gpuLock':'running','mode':'phase41-diagnostics-production-lifecycle'}))
try:
 for name in ['fresh','reload']:
  t=time.monotonic()
  with (e/f'{name}-forge.log').open('x') as log:
   p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT);(e/'full-pid.json').write_text(json.dumps({'pid':p.pid,'stage':name}))
   try:code=p.wait(timeout=1200)
   except subprocess.TimeoutExpired:p.kill();p.wait();code=-9
  row={'name':name,'exit':code,'elapsedSeconds':time.monotonic()-t};meta['stages'].append(row);print(json.dumps(row),flush=True)
  if code!=0:break
 meta['sourceUnchanged']=before=={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'Sources').rglob('*.swift')};meta['binaryUnchanged']=meta['binarySha256']==hashlib.sha256(exe.read_bytes()).hexdigest();meta['compositeArtifactWritten']=comp.exists();meta['planWritten']=plan.exists();(e/'full-run.json').write_text(json.dumps(meta,indent=2))
finally:lock.write_text(json.dumps({'gpuLock':'finished','mode':'phase41-diagnostics-production-lifecycle'}))
