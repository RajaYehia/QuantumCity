#!/usr/bin/env python

# This file creates a local network with 5 Qlients and performs MDIQKD with two of them. The outputs of the BSM
# are stored in the QOnnector's QlientKeys associated to the two Qlient sending BB84 states. The output of the file
# is printed with the number of qubit sent by Alice and the number of successful BSM. 

from QEuropeFunctions import * 

#Simulation time
simtime = 1000000

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
Alice = net.get_node("Jussieu")
Bob = net.get_node("IRIF")
Qonnector.QlientKeys[Alice.name] = []
Qonnector.QlientKeys[Bob.name] = []

#Protocol Initialisation
protocolA = SendBB84(Qonnector, Qlient_init_succ, Qlient_init_flip, Alice)
protocolA.start()    
protocolB = SendBB84(Qonnector, Qlient_init_succ, Qlient_init_flip, Bob)
protocolB.start()
protocolS = BSMProtocol(Alice, Bob, BSM_succ, Qonnector)
protocolS.start()
    
ns.sim_run(duration=simtime)

print("Qubit sent by alice : " + str(len(Alice.keylist)))
print("Successful Bell state measurements : " + str(len(Qonnector.QlientKeys[Bob.name])))
