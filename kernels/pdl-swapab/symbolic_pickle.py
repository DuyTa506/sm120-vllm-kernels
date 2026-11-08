"""Reconstruct pickle data as inert symbols; never import or invoke recorded globals.
Only use on local audit artifacts. EXT/persistent opcodes are rejected explicitly.
"""
import io
import pickle
import pickletools

class Symbol:
    def __new__(cls,*args,**kwargs):
        obj=object.__new__(cls);obj.args=args;obj.kwargs=kwargs;obj.state=None;obj.items=[];return obj
    def __init__(self,*args,**kwargs): pass
    def __setstate__(self,state): self.state=state
    def __setitem__(self,key,value): self.items.append((key,value))
    def append(self,value): self.items.append(value)
    def extend(self,values): self.items.extend(values)

class InertUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        return type('InertGlobal',(Symbol,),{'tag':(module,name)})
    def persistent_load(self,pid): raise ValueError('persistent references are not supported')

def load(data):
    for op,arg,pos in pickletools.genops(data):
        if op.name in {'EXT1','EXT2','EXT4','PERSID','BINPERSID','NEXT_BUFFER','READONLY_BUFFER'}:
            raise ValueError('unsupported external reference opcode: '+op.name)
    return InertUnpickler(io.BytesIO(data)).load()

def walk(value,seen=None):
    if seen is None:seen=set()
    if id(value) in seen:return
    seen.add(id(value));yield value
    if isinstance(value,Symbol): children=[value.args,value.kwargs,value.state,value.items]
    elif isinstance(value,dict):children=list(value.keys())+list(value.values())
    elif isinstance(value,(list,tuple,set,frozenset)):children=value
    else:return
    for v in children:yield from walk(v,seen)
