#script to visualise the output of data2pvd of MercuryDPM in paraview.
#usage: change the path below to your own path, open paraview
#Tools->Python Shell->Run Script->VisualisationScript.py
#or run paraview --script=VisualisationScript.py 

from paraview.simple import *
import os
import glob
os.chdir('/storage1/usr/people/cheng/vipr/OomphMercuryCoupling/cmake-build-debug/Drivers/OomphCouplingMultiSolid/1stElement/OneWave/pulse_trueExplicit/safe2/oneDimChain8P/CG4R')

#Load data in any order
Data = glob.glob('./oneDimChainParticle_*.vtu')

#Find the maximum timestep
maxTime = 0
for fileName in Data:
	tokens1 = fileName.split('.')
	tokens2 = tokens1[1].split('_')
	if int(tokens2[-1]) > maxTime:
		maxTime = int(tokens2[-1])
print str(maxTime)

#Create correct order of time steps
DataSorted = []
for x in range(0,maxTime+1):
	DataSorted.append('./oneDimChainParticle_' + str(x) + '.vtu')

#Load the data and visualise it in paraview
particles = XMLUnstructuredGridReader(FileName=DataSorted)
glyphP = Glyph(particles)
glyphP.GlyphType = 'Sphere'
glyphP.Scalars = 'Radius'
glyphP.Vectors = 'None'
glyphP.ScaleMode = 'scalar'
glyphP.ScaleFactor = 2
glyphP.GlyphMode = 'All Points'
Show(glyphP)

walls = XMLUnstructuredGridReader(FileName=glob.glob('./oneDimChainWall_*.vtu'))
Show(walls)

Render()
ResetCamera()
