"""Local validation reader for flat Parquet files (PLAIN / dictionary, v1 pages)."""
import struct, ctypes, json
from pathlib import Path
import numpy as np
import pandas as pd
from thrift.protocol import TCompactProtocol
from thrift.transport import TTransport
from thrift.Thrift import TType

def readval(p,t):
 if t==TType.STRUCT:
  d={}; p.readStructBegin()
  while True:
   _,tt,k=p.readFieldBegin()
   if tt==TType.STOP: break
   d[k]=readval(p,tt); p.readFieldEnd()
  p.readStructEnd(); return d
 if t in [TType.LIST,TType.SET]:
  et,n=p.readListBegin(); a=[readval(p,et) for _ in range(n)]; p.readListEnd(); return a
 if t==TType.MAP:
  kt,vt,n=p.readMapBegin(); d={readval(p,kt):readval(p,vt) for _ in range(n)};p.readMapEnd();return d
 return {TType.BOOL:p.readBool,TType.BYTE:p.readByte,TType.I16:p.readI16,TType.I32:p.readI32,TType.I64:p.readI64,TType.DOUBLE:p.readDouble,TType.STRING:p.readBinary}[t]()
def parse(b):
 tr=TTransport.TMemoryBuffer(b); d=readval(TCompactProtocol.TCompactProtocol(tr),TType.STRUCT); return d,tr._buffer.tell()
def varint(b,pos):
 v=0;s=0
 while True:
  x=b[pos];pos+=1;v|=(x&127)<<s
  if x<128:return v,pos
  s+=7

def hybrid(b,bw,n):
 out=[];i=0
 while len(out)<n:
  header,i=varint(b,i)
  if header&1:
   count=(header>>1)*8;size=(count*bw+7)//8
   bits=np.unpackbits(np.frombuffer(b[i:i+size],dtype=np.uint8),bitorder='little')
   if bw: out.extend((bits[:count*bw].reshape(count,bw) @ (1 << np.arange(bw,dtype=np.uint64))).tolist())
   else: out.extend([0]*count)
   i+=size
  else:
   count=header>>1;size=(bw+7)//8;v=int.from_bytes(b[i:i+size],'little');i+=size;out.extend([v]*count)
 return np.asarray(out[:n],dtype=np.int64)
_sn=ctypes.CDLL('libsnappy.so.1')
_sn.snappy_uncompress.argtypes=[ctypes.c_char_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t)]
def decomp(b,codec,n):
 if codec==0:return b
 if codec==1:
  out=ctypes.create_string_buffer(n);ln=ctypes.c_size_t(n);rc=_sn.snappy_uncompress(b,len(b),out,ctypes.byref(ln));assert rc==0 and ln.value==n;return out.raw
 if codec==6:
  z=ctypes.CDLL('libzstd.so.1'); z.ZSTD_decompress.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_char_p,ctypes.c_size_t]; z.ZSTD_decompress.restype=ctypes.c_size_t
  out=ctypes.create_string_buffer(n); rc=z.ZSTD_decompress(out,n,b,len(b)); assert rc==n,(rc,n);return out.raw
 if codec==2:
  import gzip;return gzip.decompress(b)
 raise NotImplementedError(('codec',codec))
def plain(b,t,n):
 if t in [1,2,4,5]:return np.frombuffer(b,dtype={1:'<i4',2:'<i8',4:'<f4',5:'<f8'}[t],count=n).tolist()
 if t==0:return np.unpackbits(np.frombuffer(b,dtype='uint8'),bitorder='little')[:n].astype(bool).tolist()
 if t==6:
  out=[];pos=0
  for _ in range(n):
   l=struct.unpack_from('<I',b,pos)[0];pos+=4;out.append(b[pos:pos+l].decode());pos+=l
  return out
 raise NotImplementedError(('plain',t))

def read_parquet(path,columns=None):
 b=Path(path).read_bytes(); assert b[:4]==b[-4:]==b'PAR1';l=struct.unpack('<I',b[-8:-4])[0];meta,_=parse(b[-8-l:-8]); schema={s[4].decode():s for s in meta[2][1:]};out={}
 for rg in meta[4]:
  for cc in rg[1]:
   m=cc[3];name=m[3][0].decode()
   if columns is not None and name not in columns:continue
   sch=schema[name]; assert len(m[3])==1 and sch.get(3,0) in [0,1]
   pos=min(m.get(11,m[9]),m[9]);stop=pos+m[7];vals=[];dictionary=None
   while pos<stop and len(vals)<m[5]:
    ph,used=parse(b[pos:]);pos+=used;raw=b[pos:pos+ph[3]];pos+=ph[3];raw=decomp(raw,m[4],ph[2])
    if ph[1]==2:
     dh=ph[7]; assert dh[2] in [0,2];dictionary=plain(raw,m[1],dh[1]);continue
    if ph[1]!=0:raise NotImplementedError(('page',ph))
    h=ph[5];num=h[1]
    if sch.get(3,0)==1:
     dl=struct.unpack('<I',raw[:4])[0];defined=hybrid(raw[4:4+dl],1,num);raw=raw[4+dl:]
    else:defined=np.ones(num,dtype=int)
    nv=int(defined.sum())
    if h[2] in [2,8]:
     ix=hybrid(raw[1:],raw[0],nv);data=[dictionary[k] for k in ix]
    elif h[2]==0:data=plain(raw,m[1],nv)
    else:raise NotImplementedError(('encoding',h[2],name))
    it=iter(data);vals.extend(next(it) if x else None for x in defined)
   assert len(vals)==m[5],(name,len(vals),m[5]);out.setdefault(name,[]).extend(vals)
 frame=pd.DataFrame(out)
 # Timestamp logical unit and pandas storage descriptors.
 for name,s in schema.items():
  if name not in frame:continue
  lt=s.get(10,{})
  if 8 in lt:
   unit=lt[8].get(2,{})
   u='ms' if 1 in unit else 'us' if 2 in unit else 'ns'
   frame[name]=pd.to_datetime(frame[name],unit=u)
 return frame
if __name__=='__main__':
 import sys
 d=read_parquet(sys.argv[1]);print(d.shape);print(d.dtypes);print(d.head().to_string())
 if len(sys.argv)>2:d.to_csv(sys.argv[2],index=False)
