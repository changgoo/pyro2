import numpy as np
import mesh.patch as patch
import matplotlib.pyplot as plt

nbuf=2
def vorticity(myg,u,v):
  vort = myg.scratch_array()
  vort.v(buf=nbuf)[:,:] = \
             0.5*(v.ip(1,buf=nbuf) - v.ip(-1,buf=nbuf))/myg.dx - \
             0.5*(u.jp(1,buf=nbuf) - u.jp(-1,buf=nbuf))/myg.dy

  return vort

def gradient(myg,w):
  grad = myg.scratch_array()
  dwdx=0.5*(w.ip(1,buf=nbuf)-w.ip(-1,buf=nbuf))/myg.dx
  dwdy=0.5*(w.jp(1,buf=nbuf)-w.jp(-1,buf=nbuf))/myg.dy
  grad.v(buf=nbuf)[:,:] = np.sqrt(dwdx**2+dwdy**2)

  return grad

def grad_vort(myg,u,v):
  gradw = myg.scratch_array()
  d2vodx2=(v.ip(1,buf=nbuf) - 2.0*v.v(buf=nbuf) + v.ip(-1,buf=nbuf))/myg.dx**2
  d2uodxdy=0.25*(u.ip_jp(1,1,buf=nbuf) - u.ip_jp(-1,1,buf=nbuf) - \
           u.ip_jp(1,-1,buf=nbuf) + u.ip_jp(-1,-1,buf=nbuf))/(myg.dx*myg.dy)
  d2uody2=(u.jp(1,buf=nbuf) - 2.0*u.v(buf=nbuf) + u.jp(-1,buf=nbuf))/myg.dy**2
  d2vodxdy=0.25*(v.ip_jp(1,1,buf=nbuf) - v.ip_jp(-1,1,buf=nbuf) - \
           v.ip_jp(1,-1,buf=nbuf) + v.ip_jp(-1,-1,buf=nbuf))/(myg.dx*myg.dy)
  gradw.v(buf=nbuf)[:,:] = np.sqrt((d2vodx2-d2uodxdy)**2+(d2vodxdy-d2uody2)**2)

  return gradw

def get_vorticity(g,d):
  u = d.get_var('x-velocity')
  v = d.get_var('y-velocity')
  w = vorticity(g,u,v)
  #gw = grad_vort(g,u,v)
  gw = gradient(g,w)

  return w.v(), gw.v()

def compare(time,res=512):
  if res == 512: 
    f1='vortex_e1_512%4.4d.pyro' % time
    f0='vortex_e0_512%4.4d.pyro' % time
  elif res == 256:
    f1='vortex_e1_256_%4.4d.pyro' % time
    f0='vortex_e0_256_%4.4d.pyro' % time
  elif res == 128:
    f1='data/vortex_e1_%4.4d.pyro' % time
    f0='data/vortex_e0_%4.4d.pyro' % time

  g1,d1 = patch.read(f1)
  g0,d0 = patch.read(f0)

  time=d1.t

  w1,gw1 = get_vorticity(g1,d1) 
  w0,gw0 = get_vorticity(g0,d0)

  dw=w1-w0
  dgw=gw1-gw0

  n1=g1.nx/4
  n2=g1.nx/4*3

  plt.clf()
  im=plt.imshow(dgw.T,origin='lower',#interpolation='nearest',
    extent=[g1.xmin,g1.xmax,g1.ymin,g1.ymax],vmin=-10,vmax=10)
  plt.plot(g1.xr,-g1.xr+1.0*time,ls=':')
  plt.plot(g1.xr,-g1.xr+2.0-1.0*time,ls=':')
  plt.xlim(0,2)
  plt.ylim(-1,1)
  plt.colorbar(im)

  dgw=dgw[n1:n2,n1:n2]
  norm=np.sum(dgw**2*g1.dx*g1.dy)

  return norm
