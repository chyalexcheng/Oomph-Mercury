# encoding: utf-8
""" Author: Hongyang Cheng <h.cheng@utwente.nl>
	This Python script plots the time evolution of
		1. total mass
		2. total linear momemtum
		3. linear momenta in the discrete and the continuum
		4. kinemtic energy
		5. potential energy
		6. total energy
	of the coupled FEM-DEM beam, and 7 variation of the numerical dissipation versus the CG width
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

def plotTwoAxes(data1,data2):
	fig, ax1 = plt.subplots()

	color = 'tab:red'
	ax1.set_xlabel('time (ms)')
	ax1.set_ylabel('FE', color=color)
	ax1.plot(data1, color=color)
	ax1.tick_params(axis='y', labelcolor=color)

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	color = 'tab:blue'
	ax2.set_ylabel('DE', color=color)  # we already handled the x-label with ax1
	ax2.plot(data2, color=color)
	ax2.tick_params(axis='y', labelcolor=color)

	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.show()

def plotEnergy(subNames,labels, savefig=False, figName='fig', noramEne=False, plotRef=False, firstIter=20, lastIter=64):
	mScale = 1e-9; jScale = 1e-9; eScale = 1e-9
	if plotRef:
		dataDEM = np.loadtxt('/'.join(subNames[0].split('/')[:-1])+'/ref/OneDimChainMercury.ene',skiprows=1)
		time0, mass_DE0, lin_mo_DE_X,lin_mo_DE_Y0,lin_mo_DE_Z, \
				 ang_mo_DE_X,ang_mo_DE_Y,ang_mo_DE_Z, \
				 kin_en_DE0, rot_en_DE0, pot_en_DE0 = dataDEM[firstIter:lastIter,DEIndex].T
		time0 *= 1e-3
		fig1 = plt.figure(1); plt.plot(time0,mScale*mass_DE0,'o',label='Full DEM',color='gray');
		fig2 = plt.figure(2); plt.plot(time0,jScale*lin_mo_DE_Y0,'o',label='Full DEM',color='gray');
		fig3 = plt.figure(3); plt.plot(time0,eScale*(kin_en_DE0+rot_en_DE0),'o',label='Full DEM',color='gray');
		fig4 = plt.figure(4); plt.plot(time0,eScale*pot_en_DE0,'o',label='Full DEM',color='gray');
		if noramEne: norm = eScale*(kin_en_DE0+rot_en_DE0+pot_en_DE0)[0]
		else: norm = 1
		fig5 = plt.figure(5); plt.plot(time0,eScale*(kin_en_DE0+rot_en_DE0+pot_en_DE0)/norm,'o',label='Full DEM',color='gray');
		fig6 = plt.figure(6); plt.plot(time0, jScale*lin_mo_DE_Y0,'o',label='Full DEM',color='gray');
	else:
		fig1 = plt.figure(1);
		fig2 = plt.figure(2);
		fig3 = plt.figure(3);
		fig4 = plt.figure(4);
		fig5 = plt.figure(5);
		fig6 = plt.figure(6);
	figsize = (8/2.54,6/2.54)

	for i,subName in enumerate(subNames):
		print('Now processing '+subName)
		dataFEM = np.loadtxt(subName+'/Trace.dat')
		dataDEM = np.loadtxt(subName+'/oneDimChain.ene',skiprows=1)
		mass_FE, lin_mo_FE_X,lin_mo_FE_Y,lin_mo_FE_Z,\
				 ang_mo_FE_X,ang_mo_FE_Y,ang_mo_FE_Z,\
				 kin_en_FE,_,pot_en_FE = dataFEM[firstIter:lastIter,4:].T
		
		time, mass_DE, lin_mo_DE_X,lin_mo_DE_Y,lin_mo_DE_Z, \
				 ang_mo_DE_X,ang_mo_DE_Y,ang_mo_DE_Z, \
				 kin_en_DE, rot_en_DE, pot_en_DE = dataDEM[firstIter:lastIter,DEIndex].T
		time *= 1e-3

		fig1 = plt.figure(1,figsize=figsize)
		plt.plot(time, mScale*(mass_FE+mass_DE),label=labels[i],color=colors[i]);
		plt.xlim(0.02) ; plt.xlabel(r'$t$ (s)'); plt.ylabel(r'$m$ (kg)'); plt.legend(loc=0); plt.tight_layout()

		fig2 = plt.figure(2,figsize=figsize)
		plt.plot(time, jScale*(lin_mo_DE_Y+lin_mo_FE_Y),label=labels[i],color=colors[i]);
		plt.xlim(0.02) ; plt.xlabel(r'$t$ (s)'); plt.ylabel(r'$j_y$ (kg'+r'$\cdot$'+'m/s)'); plt.legend(loc=0); plt.tight_layout()

		fig3 = plt.figure(3,figsize=figsize)
		plt.plot(time, eScale*(kin_en_DE+rot_en_DE+kin_en_FE),label=labels[i],color=colors[i]);
		plt.xlim(0.02) ; plt.xlabel(r'$t$ (s)'); plt.ylabel(r'$E_k$ (J)'); plt.legend(loc=0); plt.tight_layout()

		fig4 = plt.figure(4,figsize=figsize)
		plt.plot(time, eScale*(pot_en_DE+pot_en_FE),label=labels[i],color=colors[i]);
		plt.xlim(0.02) ; plt.xlabel(r'$t$ (s)'); plt.ylabel(r'$E_p$ (J)'); plt.legend(loc=0); plt.tight_layout()

		fig5 = plt.figure(5,figsize=figsize)
		if noramEne: norm = max(eScale*(kin_en_DE+rot_en_DE+pot_en_DE+kin_en_FE+pot_en_FE))
		else: norm = 1
		totalE = eScale*(kin_en_DE+rot_en_DE+pot_en_DE+kin_en_FE+pot_en_FE)
		plt.plot(time, totalE/norm,label=labels[i],color=colors[i]);
		plt.xlim(0.02) ; plt.xlabel(r'$t$ (s)'); plt.ylabel(r'$E_t$ (J)'); plt.legend(loc=0); plt.tight_layout()

		fig6 = plt.figure(6,figsize=figsize)
		ax = plt.plot(time, jScale*lin_mo_DE_Y,'o',label=labels[i]+' (DE)',color=colors[i]);
		plt.plot(time, jScale*lin_mo_FE_Y,'-',color=ax[0].get_color(),label=labels[i]+' (FE)');
		plt.xlim(0.02) ; plt.xlabel(r'$t$ (s)'); plt.ylabel(r'$j_y$ (kg'+r'$\cdot$'+'m/s)'); plt.legend(loc=0); plt.tight_layout()
	
	if savefig:
		fig1.savefig(figName+'_mass.png',dpi=300); plt.close(fig1)
		fig2.savefig(figName+'_momentum.png',dpi=300); plt.close(fig2)
		fig6.savefig(figName+'_momentum_split.png',dpi=300); plt.close(fig6)
		fig3.savefig(figName+'_ke.png',dpi=300); plt.close(fig3)
		fig4.savefig(figName+'_pe.png',dpi=300); plt.close(fig4)
		fig5.savefig(figName+'_total.png',dpi=300); plt.close(fig5)
	else: plt.show()

def plotDissipation(lists,labels, savefig=False, figName='fig', colors=[], firstIter=20, lastIter=64):
	mScale = 1e-9; jScale = 1e-9; eScale = 1e-9
	if not colors: colors = ['g','b','r']
	for m,subNames in enumerate(lists):
		CGwidths = []; DissipationRatio = []
		for i,subName in enumerate(subNames):
			dataFEM = np.loadtxt(subName+'/Trace.dat')
			dataDEM = np.loadtxt(subName+'/oneDimChain.ene',skiprows=1)
			mass_FE, lin_mo_FE_X,lin_mo_FE_Y,lin_mo_FE_Z,\
					 ang_mo_FE_X,ang_mo_FE_Y,ang_mo_FE_Z,\
					 kin_en_FE,_,pot_en_FE = dataFEM[firstIter:lastIter,4:].T
			
			time, mass_DE, lin_mo_DE_X,lin_mo_DE_Y,lin_mo_DE_Z, \
					 ang_mo_DE_X,ang_mo_DE_Y,ang_mo_DE_Z, \
					 kin_en_DE, rot_en_DE, pot_en_DE = dataDEM[firstIter:lastIter,DEIndex].T
			time *= 1e-3
			string = subName.split('/')[-1]
			CGwidth = 0 if string == 'Default' else float(string.split('CG')[-1][:-1])
			CGwidths.append(CGwidth/2.0/float(subName.split('oneDimChain')[-1].split('/')[0][:-1])**(1./3.))
			totalE = eScale*(kin_en_DE+rot_en_DE+pot_en_DE+kin_en_FE+pot_en_FE)
			DissipationRatio.append((totalE[0]-totalE[-1])/totalE[0])
		fig1 = plt.figure(1);
		if not labels:
			plt.plot(CGwidths, DissipationRatio, 'ks',markersize=2);
			plt.plot(CGwidths,DissipationRatio,'s',markerfacecolor='none', markeredgecolor='k',markersize=12);
		else:
			plt.plot(CGwidths, DissipationRatio, 's', markersize=2, label=labels[m], color=colors[m]);
			plt.plot(CGwidths,DissipationRatio,'s',markerfacecolor='none', markeredgecolor=colors[m],markersize=12)
		plt.xlim(CGwidths[0],CGwidths[-1]); plt.xlabel('CG width '+r'$(\Delta{X})$'); plt.ylabel(r'$\Delta{E_t}/E_t$'); plt.tight_layout();
		if labels: plt.legend(loc=0)
	if savefig:
		fig1.savefig(figName+'_DRatio.png',dpi=300); plt.close(fig1)
	else: plt.show()

DEIndex = np.concatenate([np.arange(0,8),np.arange(9,12)])

dirName =  "./"

plotFigs = {}
plotFigs['pulse_oneDimChain1P'] = 1
plotFigs['oneDimChainXP_Pulse'] = 2

figName = 'oneDimChainXP_Pulse'
savefig = False

# one particle per element (impulse wave load)
# pulse_oneDimChain1P
if plotFigs[figName] == 1:
	subNames = [\
				dirName + 'PulseWave/oneDimChain1P/Default',\
				dirName + 'PulseWave/oneDimChain1P/CG1R',\
				dirName + 'PulseWave/oneDimChain1P/CG2R',\
				dirName + 'PulseWave/oneDimChain1P/CG3R',\
				dirName + 'PulseWave/oneDimChain1P/CG4R',\
				dirName + 'PulseWave/oneDimChain1P/CG5R',\
				]
	colors = cm.inferno(np.linspace(0.1,0.9,len(subNames)))
	labels = [r'$\texttt{%.1f}\Delta{}\texttt{X}$'%(0.5*i) for i in [0,1,2,3,4,5]]
	plotEnergy(subNames,labels,savefig,figName,noramEne=False,plotRef=True,firstIter=0,lastIter=100)
	plotDissipation([subNames],[],savefig,figName)

# more particles per element
# oneDimChainXP_Pulse
if plotFigs[figName] == 2:
	print('plot for different numbers of particles per element')
	subNames = [\
				dirName + 'PulseWave/oneDimChain1P/CG1R',\
				dirName + 'PulseWave/oneDimChain1P/CG2R',\
				dirName + 'PulseWave/oneDimChain1P/CG3R',\
				dirName + 'PulseWave/oneDimChain8P/CG2R',\
				dirName + 'PulseWave/oneDimChain8P/CG4R',\
				dirName + 'PulseWave/oneDimChain8P/CG6R',\
				dirName + 'PulseWave/oneDimChain27P/CG3R',\
				dirName + 'PulseWave/oneDimChain27P/CG6R',\
				dirName + 'PulseWave/oneDimChain27P/CG9R',\
				dirName + 'PulseWave/oneDimChain64P/CG4R',\
				dirName + 'PulseWave/oneDimChain64P/CG8R',\
				dirName + 'PulseWave/oneDimChain64P/CG12R',\
				]
	colors = np.concatenate([cm.Greens(np.linspace(0.5,1.0,3)),\
							 cm.Greys(np.linspace(0.5,1.0,3)),\
							 cm.Blues(np.linspace(0.5,1.0,3)),\
							 cm.Reds(np.linspace(0.5,1.0,3))])
	labels = [r'$\texttt{VCoupling1P %.1fdX}$'%(0.5*i) for i in [1,2,3]]\
		   + [r'$\texttt{VCoupling8P %.1fdX}$'%(0.5*i) for i in [1,2,3]]\
		   + [r'$\texttt{VCoupling27P %.1fdX}$'%(0.5*i) for i in [1,2,3]]\
		   + [r'$\texttt{VCoupling64P %.1fdX}$'%(0.5*i) for i in [1,2,3]]
	plotEnergy(subNames,labels,savefig,figName,firstIter=0,lastIter=100)
	labels = [r'$\texttt{VCoupling8P}$', \
			  r'$\texttt{VCoupling27P}$', \
			  r'$\texttt{VCoupling64P}$']
	list1 = glob.glob(dirName + 'PulseWave/oneDimChain8P/CG*')
	list1.sort(key=len); list1 = [dirName + 'PulseWave/oneDimChain8P/Default'] + list1
	list2 = glob.glob(dirName + 'PulseWave/oneDimChain27P/CG*')
	list2.sort(key=len); list2 = [dirName + 'PulseWave/oneDimChain27P/Default'] + list2
	list3 = glob.glob(dirName + 'PulseWave/oneDimChain64P/CG*')
	list3.sort(key=len); list3 = [dirName + 'PulseWave/oneDimChain64P/Default'] + list3
	plotDissipation([list1,list2,list3],labels,savefig,figName,lastIter=83)
