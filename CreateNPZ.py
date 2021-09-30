#!/usr/bin/env python
# coding: utf-8

# In[8]:


#To get the npz file for any satellite, go to n2yo.com, chose a satellite, get the  TLE data, plug it in the TLE data parameter and run this file.
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

#%% Initialize channel paramters
wavelength = 1550e-9

#%% Initialize the satellite

# TLE data
tleMicius = ['1 25544U 98067A   21271.28780583  .00037716  00000-0  69188-3 0  9993',
'2 25544  51.6451 192.9523 0003620  33.1017 105.0724 15.48896784304579']

satMicius = channel.Satellite(tleMicius)

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

#%% Initialize the downlink channels

downSatParis = channel.SimpleDownlinkChannel(satMicius, staParis, wavelength)

downSatDelft = channel.SimpleDownlinkChannel(satMicius, staDelft, wavelength)

#%% Initialize the time array

dt = startTime = datetime(2021, 5, 15, 1, 15, 0)
endTime = datetime(2021, 5, 15, 1, 40, 0)
timeStep = timedelta(seconds = 5.)

timeList = []
while dt < endTime:
    timeList.append(dt)
    dt += timeStep
    
#%% Calculate the orbital parameters for the two channels

lenSatParis, tSatParis, elSatParis = downSatParis.calculateChannelParameters(timeList)

lenSatDelft, tSatDelft, elSatDelft = downSatDelft.calculateChannelParameters(timeList)

#%% Plot data

times = np.array([ (timeList[i]-timeList[0]).seconds  for i in range(len(timeList)) ])

plt.figure(figsize=(18,6))

plt.subplot(131)
plt.plot(times/60,elSatParis,'b')
plt.plot(times/60,elSatDelft,'r')
plt.ylim([0,90])
plt.ylabel('Elevation [degrees]')
plt.xlabel('Passage time [minutes]')
plt.legend(['Paris','Delft'])

plt.subplot(132)
plt.plot(times/60,lenSatParis/1000,'b')
plt.plot(times/60,lenSatDelft/1000,'r')
plt.ylabel('Channel length [km]')
plt.xlabel('Passage time [minutes]')
plt.legend(['Paris','Delft'])

plt.subplot(133)
plt.plot(times/60,tSatParis,'b')
plt.plot(times/60,tSatDelft,'r')
plt.ylabel('Tatm')
plt.xlabel('Passage time [minutes]')
plt.legend(['Paris','Delft'])

plt.suptitle('Micius satellite - startDate 15/05/2021 00:00')


# In[2]:


import orekit


# In[9]:


times


# In[12]:


np.savez('testISS.npz',times=times,lenSatParis=lenSatParis, tSatParis=tSatParis, elSatParis=elSatParis,lenSatDelft=lenSatDelft, tSatDelft=tSatDelft, elSatDelft=elSatDelft )


# In[ ]:




