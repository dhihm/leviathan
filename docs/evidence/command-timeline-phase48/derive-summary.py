import json,pathlib,re
b=pathlib.Path(__file__).parent
x=json.loads((b/'diagnostic-results.json').read_text())
receipt=json.loads((b/'diagnostic-run-receipt.json').read_text())
rows=[]
for f in x['fixtures']:
    s=f['summary'];n=len(f['steps']);assert n==31
    w=sum(t['wallSeconds'] for t in f['steps'])
    gpu=sum(t['gpuEnd']-t['gpuStart'] for t in f['trace']['events'])
    assert abs(gpu-s['gpuSeconds'])<1e-9
    counter=sum(t['gpuSeconds'] for t in f['steps'])
    rows.append({'fixture':f['fixture'],'commands':s['eventCount'],'kernelsPerToken':sum(t['kernelCount'] or 0 for t in f['trace']['events'])/n,'diagnosticWallMsPerStep':1000*w/n,'gpuMsPerStep':1000*gpu/n,'gpuPercentOfWall':100*gpu/w,'encodeMsPerStep':1000*s['encodeSeconds']/n,'commitCallMsPerStep':1000*s['commitCallSeconds']/n,'commitToGPUStartMsPerStep':1000*s['commitToGpuStartSeconds']/n,'GPUEndToWaitReturnMsPerStep':1000*s['gpuEndToWaitReturnSeconds']/n,'GPUEndToNextCommitMsPerStep':1000*s['gapTotalSeconds']/n,'maxGapMs':1000*(s['gapMaxSeconds'] or 0),'strictWorstRelToSerial':f['worstRelToSerial'],'strictWorstRelOnToOff':f['worstRelOnToOff'],'invalid':len(s['rejections']),'truncated':s['truncated'],'traceVsExecutorGPUSecondsDifference':gpu-counter})
def counters(side):
    text=receipt[side]['raw']['memory']['stdout']
    return {k:int(re.search('^'+k+':\\s*(\\d+)',text,re.M).group(1)) for k in ['Pageouts','Swapins','Swapouts']}
a,c=counters('before'),counters('after')
summary={'scope':'Single diagnostic trace, not speed qualification; all3fixtures strictly passed32ID/fullvocab1e-4 reference and OFF/ON gates, no output changes; previous A/A failure preserved','derivation':'GPU elapsed from rawtrace gpuEnd-gpuStart and kernels from trace kernelCount; denominator31 recorded continuation steps; subintervals overlap; driver boundaries not classified','rows':rows,'globalPagingDeltas':{k:c[k]-a[k] for k in a},'run':{k:receipt[k] for k in ['exit','elapsedSeconds','binaryUnchanged','sourceUnchanged']}}
(b/'diagnostic-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('Trace/executor GPU counter differences:',[r['traceVsExecutorGPUSecondsDifference'] for r in rows])
