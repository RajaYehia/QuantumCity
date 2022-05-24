#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 21:07:34 2021

@author: schiavon

This examples calculates the orbital parameters for the passage of a real satellite
(in this case QSS-Micius), using the two-line elements. It then calculates the
elevation, the channel length and the atmospheric losses with respect to two ground
stations, placed in Paris and Delft. These parameters can be input to the
FixedSatelliteLossModel to implement the corresponding channel on netsquid.

"""

from netsquid_freespace import channel

from datetime import datetime, timedelta

import numpy as np
from matplotlib import pyplot as plt

import os

#%% Initialize channel paramters
wavelength = 1550e-9

satName = 'Micius'
#satName = 'Starlink-1013'
#satName = 'Iridium-113'
#satName = 'Cosmos-2545'
#satName = 'OtherSat'

#%% Initialize the satellite

if satName == 'Micius':
    # Micius (QSS)
    tleMicius = ['1 41731U 16051A   21117.42314584  .00000696  00000-0  30260-4 0  9998',
                  '2 41731  97.3499  30.8507 0012844 347.0485 124.2616 15.25507799261429']
    sat = channel.Satellite(tleMicius)
elif satName == 'Starlink-1013':
    # STARLINK-1013           
    tleSL1013 = ['1 44719U 19074G   21180.08706427  .00019127  00000-0  13022-2 0  9992',
                 '2 44719  53.0534 323.7784 0001432 114.2266 245.8873 15.06322463 90483']
    sat = channel.Satellite(tleSL1013)
elif satName == 'Iridium-113':
    tleIr113 = ['1 42803U 17039A   21179.77050728  .00000070  00000-0  17932-4 0  9996',
                '2 42803  86.3976   0.3161 0002114  99.3448 260.7987 14.34217508211125']
    sat = channel.Satellite(tleIr113)
elif satName == 'Cosmos-2545':
    # COSMOS 2545 (760)       
    tleGlonass = ['1 45358U 20018A   21178.39700596 -.00000072  00000-0  00000+0 0  9993',
                  '2 45358  64.9080  15.5976 0006557 246.6968 273.8683  2.13102755  9968']
    sat = channel.Satellite(tleGlonass)
elif satName == 'OtherSat':
    # a satellite that is a 1000km away    
    tleFar = ['1 28509U 04053B   22033.30906532  .00000000  00000-0  00000-0 0  9994',
              '2 28509  63.3549 116.9036 0004692 188.5612 181.3850  2.13102259133114']
    sat = channel.Satellite(tleFar)
    
#%% Initialize the ground stations

# Paris
latParis = 48.857
longParis = 2.352
altParis = 80.

staParis = channel.GroundStation(latParis, longParis, altParis, 'Paris')

# Delft
latDelft = 52.012
longDelft = 4.357
altDelft = 0.

staDelft = channel.GroundStation(latDelft, longDelft, altDelft, 'Delft')

# Barcelona
latBarcelona = 41.383
longBarcelona = 2.183
altBarcelona = 0.

staBarcelona = channel.GroundStation(latBarcelona, longBarcelona, altBarcelona, 'Barcelona')

# Define the atmospheric model

atmModel = channel.AtmosphereTransmittanceModel(wavelength, 0, aerosolModel='NO_AEROSOLS')

#%% Initialize the downlink channels

downSatParis = channel.SimpleDownlinkChannel(sat, staParis, wavelength, atmModel =atmModel )

downSatDelft = channel.SimpleDownlinkChannel(sat, staDelft, wavelength, atmModel =atmModel)

downSatBarcelona = channel.SimpleDownlinkChannel(sat, staBarcelona, wavelength, atmModel =atmModel)


#%% Initialize the time array

if satName == 'Micius':
    # Micius
    dt = startTime = datetime(2021, 5, 15, 0, 0, 0)
    endTime = datetime(2021, 5, 15, 0, 20, 0)
elif satName == 'Starlink-1013':
    # Startlink 1013
    dt = startTime = datetime(2021, 7, 14, 2, 5, 0)
    endTime = datetime(2021, 7, 14, 2, 25, 0)
elif satName == 'Iridium-113':
    # Iridium 113
    dt = startTime = datetime(2021, 7, 30, 2, 20, 0)
    endTime = datetime(2021, 7, 30, 2, 45, 0)
elif satName == 'Cosmos-2545':
    dt = startTime = datetime(2021, 6, 30, 11, 20, 0)
    endTime = datetime(2021, 6, 30, 18, 10, 0)

elif satName == 'OtherSat':
    dt = startTime = datetime(2022, 2, 4, 9, 20, 0)
    endTime = datetime(2022, 2, 4, 15, 10, 0)
    
    
timeStep = timedelta(seconds = 10.)

timeList = []
while dt < endTime:
    timeList.append(dt)
    dt += timeStep
    
#%% Calculate the orbital parameters for the two channels

lenSatParis, tSatParis, elSatParis = downSatParis.calculateChannelParameters(timeList)

lenSatDelft, tSatDelft, elSatDelft = downSatDelft.calculateChannelParameters(timeList)

lenSatBarcelona, tSatBarcelona, elSatBarcelona = downSatBarcelona.calculateChannelParameters(timeList)

#%% Plot data

times = np.array([ (timeList[i]-timeList[0]).seconds  for i in range(len(timeList)) ])

plt.figure(figsize=(13,6))

plt.subplot(121)
plt.plot(times/60,elSatParis,'b')
plt.plot(times/60,elSatDelft,'r')
#plt.plot(times/60,elSatBarcelona,'g')
plt.ylim([0,90])
plt.ylabel('Elevation [degrees]',fontsize=15)
plt.xlabel('Passage time [minutes]',fontsize=15)
plt.tick_params(axis='both', labelsize=13)
plt.legend(['Paris','Delft'],prop={'size': 15})

plt.subplot(122)
plt.plot(times/60,lenSatParis/1000,'b')
plt.plot(times/60,lenSatDelft/1000,'r')
#plt.plot(times/60,lenSatBarcelona/1000,'g')
plt.ylabel('Channel length [km]',fontsize=15)
plt.xlabel('Passage time [minutes]',fontsize=15)
plt.tick_params(axis='both', labelsize=12)
plt.legend(['Paris','Delft'],prop={'size': 15})

#plt.subplot(133)
#plt.plot(times/60,tSatParis,'b')
#plt.plot(times/60,tSatDelft,'r')
#plt.plot(times/60,tSatBarcelona,'g')
#plt.ylabel('Tatm')
#plt.xlabel('Passage time [minutes]')
#plt.legend(['Paris','Delft','Barcelona'])

if satName == 'Micius':
    plt.suptitle('Micius satellite - startDate 15/05/2021 00:00')
elif satName == 'Starlink-1013':
    plt.suptitle('Starlink-1013 satellite - startDate 14/07/2021 02:05')
elif satName == 'Iridium-113':
    plt.suptitle('Iridium-113 satellite - startDate 30/07/2021 02:20')
elif satName == 'Cosmos-2545':
    plt.suptitle('GLONASS-760 (COSMOS 2545) satellite - startDate 30/06/2021 11:20')
    


#%% save data

dirname = './'

if satName == 'Micius':
    filename = 'satMiciusPDBurban5.npz'
    figname = 'satMiciusinfo.png'
elif satName == 'Starlink-1013':
    filename = 'satSlPDB.npz'
    figname = 'satSlinfo.png'
elif satName == 'Iridium-113':
    filename = 'satIrPDB.npz'
    figname = 'satIrinfo.png'
elif satName == 'Cosmos-2545':
    filename = 'satGloPDB.npz'
    figname = 'satGloinfo.png'
elif satName == 'OtherSat':
    filename = 'satOtherPDB.npz'
    figname = 'satOtherPDB.png'
    
plt.savefig(os.path.join(dirname,figname))

np.savez(os.path.join(dirname,filename),
          times = times,
          lenSatParis = lenSatParis,
          tSatParis = tSatParis,
          elSatParis = elSatParis,
          lenSatDelft = lenSatDelft,
          tSatDelft = tSatDelft,
          elSatDelft = elSatDelft,
          lenSatBarcelona = lenSatBarcelona,
          tSatBarcelona = tSatBarcelona,
          elSatBarcelona = elSatBarcelona)

