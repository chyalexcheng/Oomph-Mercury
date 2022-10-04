# encoding: utf-8
""" Author: Hongyang Cheng <h.cheng@utwente.nl>
	This Python script plots:
	1. The trajectories of the particles sliding on an elastic cantilever
	2. The time evolution of all energies
"""
import matplotlib.pyplot as plt
import numpy as np
import glob,os
from matplotlib.pyplot import cm
import matplotlib
import matplotlib.ticker as mtick
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

params = {'lines.linewidth': 1.5,'backend': 'ps','axes.labelsize': 18, 'axes.titlesize': 18, 'font.size': 18, 'legend.fontsize': 10.5,'xtick.labelsize': 15,'ytick.labelsize': 15,'text.usetex': True,'font.family': 'serif','legend.edgecolor': 'k','legend.fancybox':False}
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
matplotlib.rcParams.update(params)

# analytical solution for the displacement of the beam under a point load
def analyticalPrediction(pos):
	mScale = 1e-9; jScale = 1e-9; eScale = 1e-9; lScale = 1e-3; tScale = 1e-3
	# gravitation force from the particles
	F = 1.675516082e+11 * 9.81*(lScale/tScale**2)
	E = 1.0e8
	I = (0.8 /lScale)**4 / 12 
	u = F * pos**3 / (3*E*I)
	return u

def readMercuryData(f):
	fOpen = open(f,'r')
	lines = fOpen.readlines() 
	data1 = np.zeros([17,int(len(lines)/3)]) 
	data2 = np.zeros([17,int(len(lines)/3)]) 
	for i in range(int(len(lines)/3)):             
		datum1 = lines[3*i+1][:-1].split(' ')
		datum2 = lines[3*i+2][:-1].split(' ')
		datum1[1] = np.array(datum1[1]).astype(np.float)+5800
		datum2[1] = np.array(datum2[1]).astype(np.float)+5800
		data1[:,i] = np.array(datum1).astype(np.float)/1000
		data2[:,i] = np.array(datum2).astype(np.float)/1000
	return data1,data2

dirName = "./"
dirs = glob.glob(dirName+'SingleSphere/*')
savefig = False
print('Subdirectories in '+dirName+':\n'+'\n'.join([d for d in dirs]))

subNames=[
		  dirName + '/SingleSphere/Default',
		  dirName + '/SingleSphere/CG4R',
		  dirName + '/SingleSphere/CG8R',
		  dirName + '/SingleSphere/CG12R',
		  dirName + '/SingleSphere/CG20R',
		  ]
files = [subName+'/SingleSphere.data' for subName in subNames]
colors = cm.inferno(np.linspace(0,0.9,len(files)))
labels = [r'$\texttt{0d}$']+[r'$\texttt{'+'%id'%(0.5*int(subName.split('/')[-1].split('R')[0][2:]))+'}$' for subName in subNames[1:]]
figsize = (16/2.54,12/2.54)

# particle trajectories in the x-y plane
fig1 = plt.figure(7,figsize=figsize)
for n,f in enumerate(files):
	data1, data2 = readMercuryData(f)
	size = int(data1.shape[1]*0.73)
	if n == 0:
		plt.plot(data1[1,:size],np.ones(size)*data1[0,0],'o',color='gray',markevery=10,fillstyle='none',zorder=100,label='Analytical solution')
		plt.plot(data2[1,:size],np.ones(size)*data2[0,0],'o',color='gray',markevery=10,fillstyle='none',zorder=100)
	plt.plot(data1[1,:size],data1[0,:size],'-',color=colors[n],markevery=10,label=labels[n]+' (P1)')
	plt.plot(data2[1,:size],data2[0,:size],'--',color=colors[n],markevery=10,label=labels[n]+' (P2)')
plt.xlabel(r'$y$ (m)'); plt.ylabel(r'$x$ (m)'); plt.tight_layout(); plt.legend()
plt.xlim(0,12)

# particle trajectories in the y-z plane
fig2 = plt.figure(8,figsize=figsize)
ax = plt.subplot(1,1,1)
for n,f in enumerate(files):
	data1, data2 = readMercuryData(f)
	size = int(data1.shape[1]*0.73)
	if n==0: ax.plot(data1[1,:size],data1[2,0]-analyticalPrediction(data1[1,:size]),'o',color='gray',markevery=10,fillstyle='none',zorder=100,label='Analytical solution')
	ax.plot(data1[1,:size],data1[2,:size],'-',color=colors[n],markevery=10,label=labels[n]+' (P1)')
	ax.plot(data2[1,:size],data2[2,:size],'--',color=colors[n],markevery=10,label=labels[n]+' (P2)')
plt.xlabel(r'$y$ (m)'); plt.ylabel(r'$z$ (m)'); plt.tight_layout()
ax.set_xlim(0,12); ax.set_ylim(0.2,0.52)

axins = zoomed_inset_axes(ax,zoom=7,loc=3,borderpad=5)
insSize = int(size*0.55)
for n,f in enumerate(files):
	data1, data2 = readMercuryData(f)
	axins.plot(data1[1,100:insSize],data1[2,100:insSize],'-',color=colors[n],markevery=10,label=labels[n]+' (P1)')
	axins.plot(data2[1,100:insSize],data2[2,100:insSize],'--',color=colors[n],markevery=10,label=labels[n]+' (P2)')
# sub region of the original image
axins.tick_params(labelsize=14)
axins.set_xlim(1,2.25);
axins.set_ylim(0.497);
# fix the number of ticks on the inset axes
axins.yaxis.get_major_locator().set_params(nbins=3)
axins.xaxis.get_major_locator().set_params(nbins=6)

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.3",lw=0.5); plt.draw()

# discrepancy in the z-component of the displacement  
fig3 = plt.figure(9,figsize=figsize)
ax = plt.subplot(1,1,1)
for n,f in enumerate(files):
	data1, data2 = readMercuryData(f)
	size = int(data1.shape[1]*0.73)
	analSol = data1[2,0]-analyticalPrediction(data1[1,:size])
	ax.plot(data1[1,:size],data1[2,:size]-analSol,'-',color=colors[n],markevery=10,label=labels[n]+' (P1)')
	ax.plot(data2[1,:size],data2[2,:size]-analSol,'--',color=colors[n],markevery=10,label=labels[n]+' (P2)')
plt.xlabel(r'$y$ (m)'); plt.ylabel(r'$z-z_{\textnormal{ref}}$ (m)'); plt.tight_layout()
ax.set_xlim(0,12); ax.set_ylim(-0.002,0.05); 

if savefig:
	fig1.savefig('ParticlePathCG_xy.png',dpi=300)
	fig2.savefig('ParticlePathCG_xz.png',dpi=300)
	fig3.savefig('ParticlePathCGDiff_xz.png',dpi=300)
else: plt.show()

# plot kinetic, potential and total energy in time
mScale = 1e-9; jScale = 1e-9; eScale = 1e-9; lScale = 1e-3; tScale = 1e-3
DEIndex = np.concatenate([np.arange(0,8),np.arange(9,12)])
for n,subName in enumerate(subNames):
	dataFEM = np.loadtxt(subName+'/Trace.dat')
	data1, data2 = readMercuryData(subName+'/SingleSphere.data')
	size = int(data1.shape[1]*0.8)
	posP1 = data1[2,:size]-data1[2,0]
	posP2 = data2[2,:size]-data2[2,0]
	dataDEM = np.loadtxt(subName+'/SingleSphere.ene',skiprows=1)
	mass_FE, lin_mo_FE_X,lin_mo_FE_Y,lin_mo_FE_Z,\
			 ang_mo_FE_X,ang_mo_FE_Y,ang_mo_FE_Z,\
			 kin_en_FE,pot_en_FE = dataFEM[:size,5:].T
	time, mass_DE, lin_mo_DE_X,lin_mo_DE_Y,lin_mo_DE_Z, \
			 ang_mo_DE_X,ang_mo_DE_Y,ang_mo_DE_Z, \
			 kin_en_DE, rot_en_DE, pot_en_DE = dataDEM[:size,DEIndex].T
	time *= tScale
	# note pos is already scaled back to SI in function readMercuryData so we scale it back to the raw data
	pot_en_P = mass_DE/2 * (posP1+posP2)*lScale * 9.81*(lScale/tScale**2) 
	# kinetic energy on FE
	fig3 = plt.figure(9,figsize=figsize)
	plt.plot(time,jScale*(kin_en_FE),'-',color=colors[n],label=labels[n]+' (FE)'); plt.xlabel(r'$t$ (s)'); plt.ylabel('$E_k^{\mathrm{FE}}$ (J)')
	plt.xlim(0,40); plt.ylim(0); plt.tight_layout(); plt.legend()
	# kinetic energy on DE
	fig6 = plt.figure(12,figsize=figsize)
	plt.plot(time,jScale*(kin_en_DE),'--',color=colors[n],label=labels[n]+' (DE)'); plt.xlabel(r'$t$ (s)'); plt.ylabel('$E_k^{\mathrm{DE}}$ (J)')
	plt.xlim(0,40); plt.ylim(0,250); plt.tight_layout(); plt.legend()
	# potential energy on FE
	fig4 = plt.figure(10,figsize=figsize)
	plt.plot(time,jScale*(pot_en_FE),'-',color=colors[n],label=labels[n]+' (FE)'); plt.xlabel(r'$t$ (s)'); plt.ylabel('$E_p^{\mathrm{FE}}$ (J)')
	plt.xlim(0,40); plt.tight_layout(); plt.legend()
	# potential energy on DE
	fig7 = plt.figure(13,figsize=figsize)
	plt.plot(time,jScale*(pot_en_DE+pot_en_P),'--',color=colors[n],label=labels[n]+' (DE)'); plt.xlabel(r'$t$ (s)'); plt.ylabel('$E_p^{\mathrm{DE}}$ (J)')
	plt.xlim(0,40); plt.ylim(-400,25); plt.tight_layout(); plt.legend()
	# total energy on FE
	fig5 = plt.figure(14,figsize=figsize)	
	plt.plot(time,jScale*(kin_en_FE+pot_en_FE),'-',color=colors[n],label=labels[n]+' (FE)'); plt.xlabel(r'$t$ (s)'); plt.ylabel('$E_t^{\mathrm{FE}}$ (J)')
	plt.xlim(0,40); plt.tight_layout(); plt.legend()
	# total energy on DE
	fig8 = plt.figure(11,figsize=figsize)	
	plt.plot(time,jScale*(kin_en_DE+pot_en_DE+pot_en_P),'--',color=colors[n],label=labels[n]+' (DE)'); plt.xlabel(r'$t$ (s)'); plt.ylabel('$E_t^{\mathrm{DE}}$ (J)')
	plt.xlim(0,40); plt.tight_layout(); plt.legend()
	# total energy on total
	fig9 = plt.figure(15,figsize=figsize)	
	plt.semilogy(time,jScale*(kin_en_FE+kin_en_DE+pot_en_DE+pot_en_FE+pot_en_P),color=colors[n],label=labels[n]); plt.xlabel(r'$t$ (s)'); plt.ylabel('$E_t$ (J)')
	plt.xlim(0,40); plt.tight_layout(); plt.legend()

if savefig:
	fig3.savefig('ParticleBeamFE_Ek.png',dpi=300)
	fig6.savefig('ParticleBeamDE_Ek.png',dpi=300)
	fig4.savefig('ParticleBeamFE_Ep.png',dpi=300)
	fig7.savefig('ParticleBeamDE_Ep.png',dpi=300)
	fig5.savefig('ParticleBeamFE_E.png',dpi=300)
	fig8.savefig('ParticleBeamDE_E.png',dpi=300)
	fig9.savefig('ParticleBeam_E.png',dpi=300)
else: plt.show()
