import json, re
from datetime import datetime
from core.config import project_path

DEFAULT_STORE={'schema_version':1,'created_at':'','updated_at':'','next_id':1,'last_reflected_event_id':0,'reflections':[]}
def now_text(): return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
def read_store(path):
    full=project_path(path)
    if not full.exists():
        full.parent.mkdir(parents=True, exist_ok=True); data=dict(DEFAULT_STORE); data['created_at']=now_text(); data['updated_at']=now_text(); full.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8'); return data
    try: data=json.loads(full.read_text(encoding='utf-8'))
    except Exception: data=dict(DEFAULT_STORE)
    for k,v in DEFAULT_STORE.items(): data.setdefault(k,v)
    return data
def write_store(path,data):
    full=project_path(path); full.parent.mkdir(parents=True, exist_ok=True); data['updated_at']=now_text(); full.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def next_id(data):
    v=int(data.get('next_id',1)); data['next_id']=v+1; return v
def tokenize(text):
    text=str(text or '').lower(); return set(re.findall(r'[a-zA-Z0-9_]+',text)) | set(ch for ch in text if '\u4e00' <= ch <= '\u9fff')
def score_item(item,query):
    q=tokenize(query); fields=[item.get('summary',''),' '.join(item.get('stable_facts',[]) or []),' '.join(item.get('style_preferences',[]) or []),' '.join(item.get('relationship_notes',[]) or []),' '.join(item.get('persona_adjustments',[]) or [])]
    return len(q & tokenize('\n'.join(fields)))
class ReflectionMemoryStore:
    def __init__(self, reflection_file='memory_runtime/reflection_memory.json'):
        self.reflection_file=reflection_file
    def add_reflection(self, reflection):
        data=read_store(self.reflection_file); item=dict(reflection or {}); item['id']=next_id(data); item['created_at']=now_text(); item.setdefault('summary',''); item.setdefault('stable_facts',[]); item.setdefault('style_preferences',[]); item.setdefault('relationship_notes',[]); item.setdefault('persona_adjustments',[]); item.setdefault('importance',0.5); item.setdefault('source','manual_reflect')
        try: item['importance']=float(item.get('importance',0.5))
        except Exception: item['importance']=0.5
        data['reflections'].append(item)
        try:
            if item.get('last_event_id') is not None: data['last_reflected_event_id']=max(int(data.get('last_reflected_event_id',0)), int(item['last_event_id']))
        except Exception: pass
        write_store(self.reflection_file,data); return item
    def list_reflections(self, limit=20):
        items=read_store(self.reflection_file).get('reflections',[])[-int(limit):]
        if not items: return '暂无反思记忆。'
        lines=['【反思记忆】']
        for it in items:
            s=it.get('summary','')
            if len(s)>120: s=s[:120]+'...'
            lines.append(f"#{it.get('id')} importance={it.get('importance',0.5)} {it.get('created_at')}: {s}")
        return '\n'.join(lines)
    def retrieve(self, query, limit=5):
        items=read_store(self.reflection_file).get('reflections',[]); scored=[(score_item(it,query),float(it.get('importance',0.5)),it) for it in items]; scored.sort(key=lambda x:(x[0],x[1],x[2].get('id',0)), reverse=True); selected=[it for sc,imp,it in scored[:int(limit)] if sc>0]
        return selected or [it for sc,imp,it in scored[:min(3,int(limit))]]
    def format_for_prompt(self, items):
        if not items: return '暂无反思记忆。'
        lines=[]
        for it in items:
            lines.append(f"- [反思#{it.get('id')}] {it.get('summary','')}")
            for label,key in [('稳定事实','stable_facts'),('风格偏好','style_preferences'),('关系观察','relationship_notes'),('人格调整','persona_adjustments')]:
                vals=it.get(key,[]) or []
                if vals: lines.append(f"  {label}: {'；'.join(str(x) for x in vals[:3])}")
        return '\n'.join(lines)
