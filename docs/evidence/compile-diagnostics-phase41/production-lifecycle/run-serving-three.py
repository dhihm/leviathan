import hashlib,json,re,subprocess,time
from pathlib import Path
r=Path('/Users/dhihm/Documents/Codex/2026-10-02/task-5/leviathan-compile-diagnostics-phase41');e=Path('/tmp/leviathan-forge2-tooling/phase41-compiler-integrity-review/production-lifecycle');lock=e.parent.parent/'moe-root-gpu-lock.json';assert json.loads(lock.read_text())['gpuLock']=='finished'
exe=r/'.build/release/leviathan-cli';common=[str(exe),'gemma-generate','--file',str(r.parent/'models/gemma-4-26B-A4B-Q4_0/gemma-4-26B_q4_0-it.gguf'),'--graph',str(e.parent.parent/'gemma-graph.json'),'--plan',str(e/'batch-plan.json'),'--max-tokens','32','--repeat','1','--gate','--composite-plan',str(e/'composite-pass.json'),'--moe-region',str(e.parent.parent/'moe-region-root.json')]
sourceBefore={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'Sources').rglob('*.swift')};binary=hashlib.sha256(exe.read_bytes()).hexdigest();rows=[];lock.write_text(json.dumps({'gpuLock':'running','mode':'phase41-diagnostics-three-fixture-gates'}))
try:
 for name,prompt,chat in [('raw-hello','Hello',False),('en-chat','Hello, can you help me today?',True),('kr-chat','안녕하세요, 오늘 날씨는 어떤가요?',True)]:
  args=common+['--text',prompt]+(['--chat'] if chat else []);t=time.monotonic()
  with (e/f'serving-{name}.log').open('x') as log:
   p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT);(e/'serving-pid.json').write_text(json.dumps({'pid':p.pid,'fixture':name}))
   try:code=p.wait(timeout=240)
   except subprocess.TimeoutExpired:p.kill();p.wait();code=-9
  text=(e/f'serving-{name}.log').read_text();ids=re.findall(r'^generated ids: (.+)$',text,re.M);promptIDs=re.findall(r'^prompt ids: (.+)$',text,re.M)
  fallbackValues=list(map(int,re.findall(r'fallback (\d+)',text)));assert len(fallbackValues)>=9 and all(v==0 for v in fallbackValues),fallbackValues
  result={'fallbackValues':fallbackValues,'zeroFallback':True,'fixture':name,'exit':code,'elapsedSeconds':time.monotonic()-t,'gatePassed':'plan gate passed (untimed reference replay, rel < 1e-4, exact ids)' in text,'generationCompleted':'GENERATION_COMPLETED' in text,'generatedCount':len(ids[0].split()) if ids else 0,'promptTokenCount':len(promptIDs[0].split()) if promptIDs else 0,'generatedIDs':ids[0] if ids else None,'log':f'serving-{name}.log'};rows.append(result);print(json.dumps(result),flush=True)
  assert code==0 and result['gatePassed'] and result['generationCompleted'] and result['generatedCount']==32
 meta={'scope':'Actual CLI prepared execution with public verified composite installation. Separate untimed full-logit gates across rawHello and canonical English/Korean chat,32generated IDs,all262144logits,<1e-4 andexactIDs. Reference serial runner carries same verified E4/width install; no Ollama logits comparison.','fixtures':rows,'binarySha256':binary,'binaryUnchanged':binary==hashlib.sha256(exe.read_bytes()).hexdigest(),'sourceUnchanged':sourceBefore=={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'Sources').rglob('*.swift')}};(e/'serving-three-results.json').write_text(json.dumps(meta,indent=2));print('THREE_FIXTURE_SERVING_GATE_PASS',flush=True)
except BaseException as error:(e/'serving-three-failed.json').write_text(json.dumps({'fixtures':rows,'failure':repr(error)},indent=2));raise
finally:lock.write_text(json.dumps({'gpuLock':'finished','mode':'phase41-diagnostics-three-fixture-gates'}))
