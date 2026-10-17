import wave, numpy as np, sys
def envelope(path, hop=0.01):
    w=wave.open(path); sr=w.getframerate()
    a=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float32)/32768
    n=int(sr*hop); m=len(a)//n
    r=np.sqrt((a[:m*n].reshape(m,n)**2).mean(1)+1e-10)
    db=20*np.log10(r)
    k=5; db=np.convolve(db,np.ones(k)/k,mode='same')
    return db
