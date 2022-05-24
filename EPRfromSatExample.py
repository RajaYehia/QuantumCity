#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#This file simulates a satellite sending EPR pairs to two Qonnector on the ground, who transmit the photon they 
# receive to two Qlients: Bob and Hadi. The satellite data (npz file) can be created using the satcreation.py file where 
# we chose the TLE of the satellite and the atmospheric conditions. For each point of the orbit where its elevation is above 20° it averages
# on numrun times the sending of EPR pair from the satellite for a time simtime.The output are writen on a separate
# text file EPRtransmitdata.txt and plotted.

import matplotlib.pyplot as plt
from QEuropeFunctions import * 

#Satellite choice 
fic = np.load('satMiciusPDBclean.npz')

#Get chosen satellite data
times = fic['times']
elSatParis = fic['elSatParis']
elSatDelft = fic['elSatDelft']
lenSatParis = fic['lenSatParis']
lenSatDelft = fic['lenSatDelft']
tSatParis = fic['tSatParis']
tSatDelft = fic['tSatDelft']

#simulation parameters
simtime = 100000 #simulation time for each point of the orbit
numrun = 1 #number of simulation round for each point of the orbit

keysize = []
time = []
for i in range(len(times)):
    temp = 0
    temp2 = 0
    temp3 = 0
    temp4 = 0
    temp7 = 0
    if elSatParis[i] > 20 and elSatDelft[i] > 20:
        
        
        for j in range(numrun):
            ns.sim_reset()
            #Creation of a European network instance
            net2 = QEurope("Europe")
            
            #Quantum City in Paris
            net2.Add_Qonnector("QonnectorParis")
            net2.Add_Qlient("Jussieu",0.001,"QonnectorParis")
            net2.Add_Qlient("IRIF",3.01,"QonnectorParis")
            net2.Add_Qlient("Telecom",18.77,"QonnectorParis")
            net2.Add_Qlient("Chatillon",6.77,"QonnectorParis")
            net2.Add_Qlient("CEA",31.35,"QonnectorParis")
            
            #Quantum City in the Netherland
            net2.Add_Qonnector("QonnectorNetherland")
            net2.Add_Qlient("Rotterdam", 12.62,"QonnectorNetherland")
            net2.Add_Qlient("QuSoft Amsterdam",54.72,"QonnectorNetherland")
            
            #Connection of the two quantum cities via satellite
            net2.Connect_Qonnector("QonnectorParis","QonnectorNetherland", lenSatParis[i]/1000, lenSatDelft[i]/1000,
                                   tSatParis[i], tSatDelft[i], "satellite")

            
            net = net2.network
            Satellite = net.get_node("SatelliteQonnectorParisQonnectorNetherland")
            Paris = net.get_node("QonnectorParis")
            Delft =net.get_node("QonnectorNetherland")
            Bob = net.get_node("IRIF")
            Hadi = net.get_node("Rotterdam")

            Paris.QlientKeys[Satellite.name] = []
            Delft.QlientKeys[Satellite.name] = []
            Satellite.QlientKeys[Paris.name] = []
            Satellite.QlientKeys[Delft.name] = []
            Bob.keylist = []
            Hadi.keylist = []
            
            #protocol to send EPR pairs from the satellite at a rate f_EPR and with success probability EPR_succ
            ProtocolS = SendEPR(Paris, Delft, EPR_succ, Satellite)
            ProtocolS.start()
           
            #Transmission to the Qlients
            protocolS1 = TransmitProtocol(Satellite, Bob, switch_succ, Paris)
            protocolS1.start()
            protocolS2 = TransmitProtocol(Satellite, Hadi, switch_succ, Delft)
            protocolS2.start()
            
            #Protocol to receive photons with success probability Qonnector_meas_succ and flipping probability Qonnector_meas_flip.
            #This protocol measure randomly the arriving photon in the X or Z basis and stor the output in list QlientKey of the qonnectors
            protocolA = ReceiveProtocol(Paris, Qonnector_meas_succ, Qonnector_meas_flip,False, Bob)
            protocolA.start()

            protocolB = ReceiveProtocol(Delft, Qonnector_meas_succ, Qonnector_meas_flip,False, Hadi)
            protocolB.start()
            stat =ns.sim_run(duration=simtime)
            
            #Adding dark counts
            addDarkCounts(Bob.keylist, pdarkbest, int(i/Qonnector_init_time))
            addDarkCounts(Hadi.keylist, pdarkbest, int(i/Qonnector_init_time))
            
            #Sifting: we only keep the qubit Bob and Hadi received from the same pair
            L = Sifting(Bob.keylist,Hadi.keylist) 
            temp = temp + len(L)
            temp7 = temp7 + len(Bob.keylist)
            if len(Satellite.QlientKeys[Paris.name]) !=0 :
                temp4 = len(Satellite.QlientKeys[Paris.name])
                temp2 =  temp2 +len(L)/len(Satellite.QlientKeys[Paris.name])
                if L !=[]:
                    temp3 = temp3 + estimQBEREPR(L)
            
    print(i)
    keysize.append(temp/numrun)

    time.append(times[i])

    if elSatParis[i] > 20:
        somdata = open("EPRtransmitdata.txt","a")
        somdata.write (str(times[i]) + "min: \n" ) 
        somdata.write("EPR sent by the satellite: " + str(temp4/numrun)+"\n")
        somdata.write("photon received by Hadi and Bob: "+ str(temp/numrun)+"\n")
        somdata.write("photon received by Bob: "+ str(temp7/numrun)+"\n")
        somdata.write("Throughput: "+ str(temp2/numrun)+"\n")
        somdata.write("QBER: " + str(temp3/numrun) + "\n")
        somdata.close()
    
plt.figure(figsize=(15,9)) 
plt.plot(time,keysize,label="Number of EPR pairs successfully measured in Paris and Delft")

plt.xlabel('Time (min) ',size=20)
plt.ylabel('Raw Sfited Key size (bit)',size=20)
plt.legend(loc='upper right',prop={'size':20})

plt.show()

