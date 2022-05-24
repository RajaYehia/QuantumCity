#!/usr/bin/env python

#This file creates a local network with one Qonnector and 4 Qlient then simulates the creation and sending of a 
# 4 qubits GHZ state. The qubits received are stored in each Qlient's keylist.
# The output printed are the number of successful GHZ reception the rate and the QBER.

import matplotlib.pyplot as plt
from QEuropeFunctions import * 

#Simulation time
simtime = 100000

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

#Initialisation of the nodes
net = net2.network
Qonnector=net.get_node("QonnectorParis")
Alice = net.get_node("Chatillon")
Bob = net.get_node("Telecom")
Charlie = net.get_node("Jussieu")
Dina = net.get_node("IRIF")
Alice.keylist=[]
Bob.keylist=[]
Charlie.keylist=[]
Dina.keylist=[]

#Initialisation of the protocol
GHZProtocol = SendGHZ4(Alice, Bob, Charlie, Dina, GHZ4_succ, Qonnector)
GHZProtocol.start()

protocolA = ReceiveProtocol(Qonnector, Qlient_meas_succ, Qlient_meas_flip, False, Alice)
protocolA.start()

protocolB = ReceiveProtocol(Qonnector, Qlient_meas_succ, Qlient_meas_flip, False, Bob)
protocolB.start()

protocolC = ReceiveProtocol(Qonnector, Qlient_meas_succ, Qlient_meas_flip, False, Charlie)
protocolC.start() 
        
protocolD = ReceiveProtocol(Qonnector, Qlient_meas_succ, Qlient_meas_flip,False, Dina)
protocolD.start()

#Simulation starting
stat = ns.sim_run(duration=simtime)

#Adding dark count for each Qlient
addDarkCounts(Alice.keylist, pdarkworst, int(simtime/GHZ4_time))
addDarkCounts(Bob.keylist, pdarkworst, int(simtime/GHZ4_time))
addDarkCounts(Charlie.keylist, pdarkbest, int(simtime/GHZ4_time))
addDarkCounts(Dina.keylist, pdarkbest, int(simtime/GHZ4_time))

#Sifting to keep the qubit from the same GHZ state
Lres=Sifting4(Alice.keylist,Bob.keylist,Charlie.keylist,Dina.keylist)



print("Number of qubits received by the four Qlients: " +str(len(Lres)) )
print("GHZ4 sharing rate : " + str(len(Lres)/(simtime*1e-9))+" GHZ4 per second")

print("QBER : "+str(estimQBERGHZ4(Lres)))

