#!/usr/bin/env python
# This file creates a local network with one Qonnector and one Qlient separated by a distance 'distance'. They perform the BB84 protocol with the Qonnector sending photons to the Qlient through an 
#  optical fiber. The simulation goes for a time given by 'simtime' and the output printed is the number of qubits sent and received, the raw key rate, the throughput and the QBER.

from QEuropeFunctions import *

distance = 5 #distance between the Qlient and the Qonnector (km)

#simulation time (ns)
simtime = 100000 

#Simulation
ns.sim_reset()
net2 = QEurope("Europe")
net2.Add_Qonnector("QonnectorParis")
net2.Add_Qlient("Bob",distance,"QonnectorParis")


net = net2.network
Alice = net.get_node("QonnectorParis")
Bob =net.get_node("Bob")

Alice.QlientKeys[Bob.name] = []
Bob.keylist = []

ProtocolS = SendBB84(Bob, Qonnector_init_succ, Qonnector_init_flip, Alice)
ProtocolS.start()

protocolA = ReceiveProtocol(Alice, Qlient_meas_succ, Qlient_meas_flip,True, Bob)
protocolA.start()
            
addDarkCounts(Bob.keylist, pdarkbest, int(simtime/Qonnector_init_time))

stat =ns.sim_run(duration=simtime)

Lres = Sifting(Bob.keylist,Alice.QlientKeys[Bob.name])

print("Number of qubits sent by Alice (Qonnector): " +str(len(Alice.QlientKeys[Bob.name])) )
print("Number of qubits received by Bob (Qlient): " +str(len(Bob.keylist)) )
print("Raw key rate : " + str(len(Bob.keylist)/(simtime*1e-9))+" bits per second")
print("Throughput : "+str(len(Bob.keylist)/len(Alice.QlientKeys[Bob.name])) + " bits per channel use")
print("QBER : "+str(estimQBER(Lres)))

