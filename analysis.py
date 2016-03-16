import numpy as np
import mesh.patch as patch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,SymLogNorm
from astropy.io import ascii

nbuf=2
def vorticity(myg,u,v):
  vort = myg.scratch_array()
  vort.v(buf=nbuf)[:,:] = \
             0.5*(v.ip(1,buf=nbuf) - v.ip(-1,buf=nbuf))/myg.dx - \
             0.5*(u.jp(1,buf=nbuf) - u.jp(-1,buf=nbuf))/myg.dy

  return vort

def gradient(myg,w):
  grad = myg.scratch_array()
  dwdx = myg.scratch_array()
  dwdy = myg.scratch_array()
  dwdx.v(buf=nbuf)[:,:]=0.5*(w.ip(1,buf=nbuf)-w.ip(-1,buf=nbuf))/myg.dx
  dwdy.v(buf=nbuf)[:,:]=0.5*(w.jp(1,buf=nbuf)-w.jp(-1,buf=nbuf))/myg.dy

  return dwdx,dwdy

def grad_vort(myg,u,v):
  gradw = myg.scratch_array()
  d2vodx2=(v.ip(1,buf=nbuf) - 2.0*v.v(buf=nbuf) + v.ip(-1,buf=nbuf))/myg.dx**2
  d2uodxdy=0.25*(u.ip_jp(1,1,buf=nbuf) - u.ip_jp(-1,1,buf=nbuf) - \
           u.ip_jp(1,-1,buf=nbuf) + u.ip_jp(-1,-1,buf=nbuf))/(myg.dx*myg.dy)
  d2uody2=(u.jp(1,buf=nbuf) - 2.0*u.v(buf=nbuf) + u.jp(-1,buf=nbuf))/myg.dy**2
  d2vodxdy=0.25*(v.ip_jp(1,1,buf=nbuf) - v.ip_jp(-1,1,buf=nbuf) - \
           v.ip_jp(1,-1,buf=nbuf) + v.ip_jp(-1,-1,buf=nbuf))/(myg.dx*myg.dy)
  gradw.v(buf=nbuf)[:,:] = np.sqrt((d2vodx2-d2uodxdy)**2+(d2vodxdy-d2uody2)**2)
  dwdx=d2vodx2-d2uodxdy
  dwdy=d2vodxdy-d2uody2

  return dwdx,dwdy

def get_vorticity(g,d):
  u = d.get_var('x-velocity')
  v = d.get_var('y-velocity')
  w = vorticity(g,u,v)
  #gw = grad_vort(g,u,v)
  gwx,gwy = gradient(g,w)

  return w.v(), gwx.v(), gwy.v()

def compare(time,res=512,base='/Users/cgkim/Dropbox/pyro/data/',fileout=False):
  if res == 512: 
    f1='vortex_e1_512%4.4d.pyro' % time
    f0='vortex_e0_512%4.4d.pyro' % time
  elif res == 256:
    f1='vortex_e1_256_%4.4d.pyro' % time
    f0='vortex_e0_256_%4.4d.pyro' % time
  elif res == 128:
    f1='vortex_e1_%4.4d.pyro' % time
    f0='vortex_e0_%4.4d.pyro' % time

  g1,d1 = patch.read(base+f1)
  g0,d0 = patch.read(base+f0)

  time=d1.t

  w1,gwx1,gwy1 = get_vorticity(g1,d1) 
  w0,gwx0,gwy0 = get_vorticity(g0,d0)

  dw=w1-w0
  dgwx=gwx1-gwx0
  dgwy=gwy1-gwy0
  dgw=np.sqrt(dgwx**2+dgwy**2)

  n1=g1.nx/4
  n2=g1.nx/4*3
#  print dgw.shape

  plt.clf()
  y2d=g1.y2d[g1.ilo:g1.ihi+1,g1.jlo:g1.jhi+1] 
  x2d=g1.x2d[g1.ilo:g1.ihi+1,g1.jlo:g1.jhi+1] 
  idx1=y2d < (-x2d+0.5+0.4*time)
  idx2=y2d > (-x2d+1.5-0.4*time)
  idx3=y2d > (+x2d+0.6*time)
  idx4=y2d < (+x2d-2.0-0.6*time)
  dgw[idx1]=0.
  dgw[idx2]=0.
  dgw[idx3]=0.
  dgw[idx4]=0.
  upper=y2d > -x2d
  lower=y2d < -x2d

#  x1,y1= x2d[upper][np.argmax(dgw[upper])],y2d[upper][np.argmax(dgw[upper])]
#  x2,y2= x2d[lower][np.argmax(dgw[lower])],y2d[lower][np.argmax(dgw[lower])]
#  print x1,y1,x2,y2
#  print np.sqrt((x1-x2)**2+(y1-y2)**2),g1.dx
  x1,y1= x2d.flatten()[np.argmax(dgw)],y2d.flatten()[np.argmax(dgw)]
  nres=np.sqrt((x1-1.0)**2+(y1-0.0)**2)/g1.dx

  im=plt.imshow(dgw.T,origin='lower',interpolation='nearest',
    extent=[g1.xmin,g1.xmax,g1.ymin,g1.ymax],
    norm=LogNorm(vmin=1.e-2,vmax=1.e2))
  plt.plot(g1.xr,-g1.xr+0.5+0.4*time,ls=':',color='w')
  plt.plot(g1.xr,-g1.xr+1.5-0.4*time,ls=':',color='w')
  plt.plot(g1.xr,+g1.xr+0.6*time,ls=':',color='w')
  plt.plot(g1.xr,+g1.xr-2.0-0.6*time,ls=':',color='w')
  plt.xlim(0,2)
  plt.ylim(-1,1)
  plt.colorbar(im)
  plt.scatter(x1,y1,marker='*')
  plt.axhline(0,ls=':')
  plt.axvline(1,ls=':')

  norm=np.sum(dgw**2*g1.dx*g1.dy)

  if fileout:
    if time == 0.0: 
      fp=open('norm_%d.txt' % res,'w')
    else:
      fp=open('norm_%d.txt' % res,'a')
    fp.write("%15.5e %15.5e %15.5e\n" % (time,norm,nres))
    fp.close() 
  print time,norm,nres
  return time,norm

def plot_norm():
  datah=ascii.read("norm_512.txt") 
  datai=ascii.read("norm_256.txt") 
  datal=ascii.read("norm_128.txt") 

  for d in [datal,datai,datah]:
    t=d['col1']
    n=d['col2']
    nres=d['col3'] <5
    tmax=t[nres].min()
    print tmax
    l,=plt.plot(t,n)
    plt.axvline(tmax,ls=':',color=l.get_color())
  plt.yscale('log')
