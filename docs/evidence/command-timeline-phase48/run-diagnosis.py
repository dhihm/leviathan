import datetime,hashlib,json,pathlib,subprocess,time
b=pathlib.Path(__file__).parent
root=pathlib.Path('<workspace>/leviathan-command-timeline-phase48')
model=root.parent/'models/gemma-4-26B-A4B-Q4_0/gemma-4-26B_q4_0-it.gguf'
common=b.parent
lock=common/'moe-root-gpu-lock.json'
assert json.loads(lock.read_text())['gpuLock']=='finished', 'tracked owned GPU task not finished'
exe=b/'diagnose'
argv=[str(exe),str(model),str(common/'gemma-graph.json'),str(common/'moe-region-root.json'),str(common/'phase41-compiler-integrity-review/production-lifecycle/serving-three-results.json'),str(b/'diagnostic-results.json')]
assert not (b/'diagnostic-results.json').exists(), 'retain previous evidence; no overwrite or adaptive rerun'
def observe(label):
    commands={'memory':['vm_stat'],'thermal':['pmset','-g','therm'],'power':['pmset','-g','batt'],'load':['uptime']}
    values={}
    for key,args in commands.items():
        p=subprocess.run(args,capture_output=True,text=True);values[key]={'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    return {'label':label,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'raw':values}
meta={'scope':'Retained fixed three-fixture diagnostic, not a performance gain evaluation','argv':argv,'binarySha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'before':observe('outside-before'),'startedUtc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sourceBefore':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'Sources').rglob('*.swift')}}
lock.write_text(json.dumps({'gpuLock':'running','mode':'phase48-command-timeline-diagnosis'}))
t=time.monotonic()
try:
    with (b/'diagnostic-run.log').open('x') as log:
        child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT)
        (b/'owned-child.json').write_text(json.dumps({'pid':child.pid,'kind':'owned child from this exact invocation','argv':argv},indent=2)+'\n')
        try:meta['exit']=child.wait(timeout=600)
        except subprocess.TimeoutExpired:child.kill();child.wait();meta['exit']=-9;meta['timeout']=True
    meta['elapsedSeconds']=time.monotonic()-t
    meta['after']=observe('outside-after')
    meta['binaryUnchanged']=meta['binarySha256']==hashlib.sha256(exe.read_bytes()).hexdigest()
    meta['sourceUnchanged']=meta['sourceBefore']=={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'Sources').rglob('*.swift')}
    meta['completedUtc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (b/'diagnostic-run-receipt.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({k:meta[k] for k in ['exit','elapsedSeconds','binaryUnchanged','sourceUnchanged','completedUtc']}),flush=True)
finally:
    lock.write_text(json.dumps({'gpuLock':'finished','mode':'phase48-command-timeline-diagnosis'}))
raise SystemExit(meta['exit'])
