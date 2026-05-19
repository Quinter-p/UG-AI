import json, re, urllib.request
from storage.event_log_store import EventLogStore
from storage.reflection_memory_store import ReflectionMemoryStore

def extract_json(text):
    text=str(text or '').strip(); text=re.sub(r'```json\s*','',text,flags=re.I).replace('```','').strip()
    if '</think>' in text: text=text.split('</think>',1)[-1].strip()
    s=text.find('{'); e=text.rfind('}')
    if s>=0 and e>s: text=text[s:e+1]
    return json.loads(text)
def get_base(config):
    r=config.get('reflection',{}) or {}
    if r.get('base_url'): return str(r['base_url']).rstrip('/')
    for k in ['ollama','llm','model']:
        v=config.get(k)
        if isinstance(v,dict) and v.get('base_url'): return str(v['base_url']).rstrip('/')
    return 'http://127.0.0.1:11434'
def get_model(config):
    r=config.get('reflection',{}) or {}
    if r.get('model'): return str(r['model'])
    for k in ['ollama','llm','model']:
        v=config.get(k)
        if isinstance(v,dict) and v.get('model'): return str(v['model'])
        if isinstance(v,str) and ':' in v: return v
    return ''
def call_ollama_json(config,messages):
    r=config.get('reflection',{}) or {}; model=get_model(config)
    if not model: raise RuntimeError('未找到 reflection.model 或主模型 model 配置。')
    payload={'model':model,'messages':messages,'stream':False,'options':{'temperature':float(r.get('temperature',0.2)),'top_p':float(r.get('top_p',0.8)),'num_predict':int(r.get('num_predict',700))}}
    if r.get('num_ctx'): payload['options']['num_ctx']=int(r['num_ctx'])
    req=urllib.request.Request(get_base(config)+'/api/chat',data=json.dumps(payload,ensure_ascii=False).encode('utf-8'),headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=float(r.get('timeout',30))) as resp: raw=resp.read().decode('utf-8',errors='replace')
    return extract_json((json.loads(raw).get('message') or {}).get('content',''))
def normalize_reflection(obj,source='manual_reflect',last_event_id=None):
    item=dict(obj or {}); item.setdefault('summary',''); item.setdefault('stable_facts',[]); item.setdefault('style_preferences',[]); item.setdefault('relationship_notes',[]); item.setdefault('persona_adjustments',[]); item.setdefault('importance',0.5); item['source']=source
    if last_event_id is not None: item['last_event_id']=last_event_id
    for k in ['stable_facts','style_preferences','relationship_notes','persona_adjustments']:
        v=item.get(k,[]); item[k]=[v] if isinstance(v,str) and v.strip() else ([str(x) for x in v if str(x).strip()] if isinstance(v,list) else [])
    try: item['importance']=max(0.0,min(1.0,float(item.get('importance',0.5))))
    except Exception: item['importance']=0.5
    return item
def build_reflection_messages(events_text):
    system='你是 AI Agent 的反思模块，只输出 JSON。你要从最近事件中提炼稳定结论，不要记录临时闲聊。不要把今天吃了什么、刚刚问了什么这种短期上下文写入反思。应该提取：主人稳定偏好、AI 回复风格改进点、关系变化、长期设定一致性。只输出一个 JSON object，不要 markdown。'
    user={'recent_events':events_text,'output_schema':{'summary':'一段中文总结，80-200字','stable_facts':['稳定事实，不确定就留空'],'style_preferences':['主人对回复风格的稳定偏好'],'relationship_notes':['关于主人/熟人/敌人的长期关系观察'],'persona_adjustments':['AI 以后说话需要遵守的长期调整'],'importance':'0.0-1.0'},'rules':['只保留稳定趋势，不保留偶然闲聊。','不要夸大，不要编造没出现过的事实。','如果事件不足以反思，summary 写“近期事件不足以形成稳定反思”，importance 给 0.1。']}
    return [{'role':'system','content':system},{'role':'user','content':json.dumps(user,ensure_ascii=False)}]
def run_reflection_now(config,source='manual_reflect'):
    rcfg=config.get('reflection',{}) or {}; ecfg=config.get('event_log',{}) or {}; ev=EventLogStore(ecfg.get('event_file','memory_runtime/event_log.jsonl')); rs=ReflectionMemoryStore(rcfg.get('reflection_file','memory_runtime/reflection_memory.json')); limit=int(rcfg.get('event_limit',30)); events=ev.read_recent(limit)
    if not events: return {'ok':False,'message':'暂无事件日志，无法反思。'}
    saved=rs.add_reflection(normalize_reflection(call_ollama_json(config,build_reflection_messages(ev.compact_for_reflection(limit))),source,events[-1].get('id')))
    return {'ok':True,'message':f"已生成反思记忆 #{saved.get('id')}：{saved.get('summary','')}",'reflection':saved}
def reflection_memory_load_node(state):
    cfg=state.get('config') or {}; rcfg=cfg.get('reflection',{}) or {}
    if not rcfg.get('enabled',True): return {'reflection_memory_items':[],'reflection_memory_text':'反思记忆未启用。'}
    store=ReflectionMemoryStore(rcfg.get('reflection_file','memory_runtime/reflection_memory.json')); rel=state.get('relationship_state') or {}; query=' '.join([str(state.get('clean_text') or state.get('raw_message') or ''),str((state.get('identity') or {}).get('name','')),str(rel.get('name','')),str(rel.get('role',''))]); items=store.retrieve(query,int(rcfg.get('prompt_reflection_limit',5)))
    return {'reflection_memory_items':items,'reflection_memory_text':store.format_for_prompt(items)}
def reflection_maybe_node(state):
    cfg=state.get('config') or {}; rcfg=cfg.get('reflection',{}) or {}
    if not rcfg.get('enabled',True) or not rcfg.get('auto_enabled',False): return {}
    ecfg=cfg.get('event_log',{}) or {}; ev=EventLogStore(ecfg.get('event_file','memory_runtime/event_log.jsonl')); data=__import__('storage.reflection_memory_store',fromlist=['read_store']).read_store(rcfg.get('reflection_file','memory_runtime/reflection_memory.json'))
    if ev.count()-int(data.get('last_reflected_event_id',0)) < int(rcfg.get('auto_every_events',20)): return {}
    try: return {'reflection_status':run_reflection_now(cfg,'auto_reflect').get('message','')}
    except Exception as e: return {'reflection_status':f'自动反思失败：{type(e).__name__}: {e}'}
