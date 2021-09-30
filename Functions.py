#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import netsquid as ns

import netsquid.components.instructions as instr
import netsquid.components.qprogram as qprog
import random 
from scipy.stats import bernoulli
import logging
import math
import numpy as np

from netsquid.components import Channel, QuantumChannel, QuantumMemory, ClassicalChannel
from netsquid.components.models.qerrormodels import FibreLossModel, DepolarNoiseModel, DephaseNoiseModel
from netsquid.nodes import Node, DirectConnection
from netsquid.nodes.connections import Connection
from netsquid.protocols import NodeProtocol
from netsquid.components.models import DelayModel
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel
from netsquid.components import QuantumMemory
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.nodes.network import Network
from netsquid.qubits import ketstates as ks
from netsquid.protocols.protocol import Signals
from netsquid.components.qprocessor import PhysicalInstruction
from netsquid.qubits import qubitapi as qapi
from netsquid.components.clock import Clock

from lossmodel import FreeSpaceLossModel, FixedSatelliteLossModel
#import channel

#from datetime import datetime, timedelta

import matplotlib.pyplot as plt

#Qlient parameters
f_qubit_qlient = 80e6 #Qubit creation attempt frequency
Qlient_init_time = math.ceil(1e9/f_qubit_qlient) #time to create |0> in a Qlient node in ns
Best_init_succ = 0.008 #Probability that a a qubit creation succeed
Worst_init_succ = 5e-3
Best_init_flip = 0#probability that a qubit is flipped at its creation
Worst_init_flip = 0.01
Best_measurement_succ=0.95 #Probability that a measurement succeeds
Worst_measurement_succ = 0.85
Mid_measurement_succ = 0.9
Best_meas_flip = 1e-5 #Probability that the measurement outcome is flipped by the detectors 
Worst_meas_flip = 1e-2 


#Qonnector parameters
Max_Qlient = 5 #Number of simultaneous link that the Qonnector can create 
f_qubit_qonn = 80e6 #Qubit creation attempt frequency in MHz
Qonnector_init_time = math.ceil(1e9/f_qubit_qonn) #time to create |0> in a Qonnector node in ns
Qonnector_init_succ = 0.008 #Probability that a qubit creation succeeds
Qonnector_init_flip = 0
Qonnector_meas_succ=0.95 #Probability that a measurement succeeds
Qonnector_meas_flip = 1e-5 #Probability that the measurement outcome is flipped by the detectors 
switch_succ=0.9 #probability that transmitting a qubit from a qlient to another succeeds
BSM_succ = 0.36 #probability that a Bell state measurement of 2 qubits succeeds
EPR_succ=0.01 #probability that an initialisation of an EPR pair succeeds
f_EPR = 80e6 #EPR pair creation attempt frequency in MHz
EPR_time = math.ceil(1e9/f_EPR) # time to create a bell pair in a Qonnector node (ns)
f_GHZ = 8e6 #GHZ state creation attempt frequency in MHz
GHZ3_time = math.ceil(1e9/f_GHZ) #time to create a GHZ3 state (ns)
GHZ3_succ = 2.5e-3 #probability that creating a GHZ3 state succeeds
GHZ4_time = math.ceil(1e9/f_GHZ) #time to create a GHZ4 state (ns)
GHZ4_succ = 3.6e-3 #probability that creating a GHZ4 state succeeds
GHZ5_time = math.ceil(1e9/f_GHZ) #time to create a GHZ5 state (ns)
GHZ5_succ = 9e-5 #probability that creating a GHZ5 state succeeds

#Dark Counts parameter
DCRateBest = 100
DCRateWorst = 1000
DetectGateBest = 1e-10
DetectGateWorst = 5e-10
pdarkbest = DCRateBest*DetectGateBest
pdarkworst = DCRateWorst*DetectGateWorst


#Network parameter
fiber_coupling = 0.9 #Fiber coupling efficiency
fiber_loss=0.18 #Loss in fiber in dB/km
fiber_dephasing_rate = 0.02 #dephasing rate in the fiber (Hz)

#Free space channel parameter
W0 = 0.1
rx_aperture_freespace = 1
Cn2_freespace = 0#1e-15
wavelength = 850*10e-9
c = 299792.458 #speed of light in km/s
Tatm = 1

#Satellite to Ground channel parameters
txDiv = 10e-6
sigmaPoint = 0.5e-6
rx_aperture_sat = 1
Cn2_sat = 0

qonnector_physical_instructions = [
    PhysicalInstruction(instr.INSTR_INIT, duration=Qonnector_init_time),
    PhysicalInstruction(instr.INSTR_H, duration=1, parallel=True, topology=[0]),
    PhysicalInstruction(instr.INSTR_X, duration=1, parallel=True, topology=[0]),
    PhysicalInstruction(instr.INSTR_Z, duration=1, parallel=True, topology=[0]),
    PhysicalInstruction(instr.INSTR_S, duration=1, parallel=True, topology=[0]),
    PhysicalInstruction(instr.INSTR_I, duration=1, parallel=True, topology=[0]),
    PhysicalInstruction(instr.INSTR_CNOT, duration=4, parallel=True),
    PhysicalInstruction(instr.INSTR_MEASURE, duration=1, parallel=True, topology=[0,1]),
    PhysicalInstruction(instr.INSTR_MEASURE_BELL, duration = 1, parallel=True),
    PhysicalInstruction(instr.INSTR_SWAP, duration = 1, parallel=True)
]

qlient_physical_instructions = [
    PhysicalInstruction(instr.INSTR_INIT, duration=Qlient_init_time),
    PhysicalInstruction(instr.INSTR_H, duration=1, parallel=True, topology=[0]),
    PhysicalInstruction(instr.INSTR_X, duration=1, parallel=True, topology=[0]),
    PhysicalInstruction(instr.INSTR_Z, duration=1, parallel=True, topology=[0]),
    PhysicalInstruction(instr.INSTR_S, duration=1, parallel=True, topology=[0]),
    PhysicalInstruction(instr.INSTR_I, duration=1, parallel=True, topology=[0]),
    PhysicalInstruction(instr.INSTR_MEASURE, duration=1, parallel=False, topology=[0])
]

class Qlient_node(Node):
    """A Qlient node
    
    Parameters:
    name: name of the Qlient
    phys_instruction: list of physical instructions for the Qlient
    keylist: list of bits for QKD
    ports: list of two ports: one to send to Qonnector and one to receive
    """
    
    def __init__(self, name, phys_instruction, keylist=None,listports=None):
        super().__init__(name=name)
        qmem = QuantumProcessor("QlientMemory{}".format(name), num_positions=1,
                                phys_instructions=phys_instruction)
        self.qmemory = qmem
        self.keylist=keylist
        self.listports=listports
        
class Qonnector_node(Node):
    """A Qonnector node
    
    Parameters:
    QlientList: List of connected Qlients
    QlientPorts: Dictionnary of the form {Qlient: [port_to_send, port_to_receive]}
    QlientKeys : Dictionnary for QKD of the form {Qlient: [key]}
    """
    
    def __init__(self, name,QlientList=None,
                  QlientPorts=None,QlientKeys=None):
        super().__init__(name=name)
        self.QlientList = QlientList
        self.QlientPorts = QlientPorts
        self.QlientKeys = QlientKeys
        

class QEurope():
    
    def __init__(self,name):
        # Initialisation of a Quantum network
        self.network = Network(name)
        self.name=name
    
    def Add_Qonnector(self, qonnectorname):
        #Method to add a Qonnector to the network
        
        ### Parameters ###
        # qonnectorname: name tof the Qonnector to add (str)
        
        Qonnector = Qonnector_node(qonnectorname, QlientList=[],QlientPorts={},QlientKeys={})
        self.network.add_node(Qonnector)


    def Add_Qlient(self, qlientname, distance, qonnectorto):
    
        # Method to add a Qlient to the network. It creates a Quantum Processor at the Qonnector qonnectorto
        # that is linked to the new Qlient.
        
        ### Parameters ###
        # network: network to add the qlient to (specified by a Qonnector)
        # qlientname: name of the qlient to add (str)
        # distance: distance from the QOnnector to the new node in km
        # list_ports: list of : Qlient port to receive from Qonnector, Qlient port to send to Qonnector,
        #             Qonnector port to receive from Qlient, Qlient port to send to Qlient
        # phys_instr: physical instructions for the Qlient
        
        network = self.network
        # Check that the Qonnector has space for the new qlient
        Qonnector = network.get_node(qonnectorto)
        if len(Qonnector.QlientList)==Max_Qlient:
            raise ValueError("You have reached the maximum Qlient capacity for this Qonnector.")
        
        #creates a qlient and adds it to the network
        Qlient = Qlient_node(qlientname,qlient_physical_instructions,keylist=[],listports=[])
        network.add_node(Qlient) 
        
        #Create quantum channels and add them to the network
      
        qchannel1 = QuantumChannel("QuantumChannelSto{}".format(qlientname),length=distance, delay=1,
                                   models={"quantum_loss_model": FibreLossModel(p_loss_init=1-fiber_coupling,
                                                                                p_loss_length=fiber_loss),
                                          "quantum_noise_model":DephaseNoiseModel(dephase_rate = fiber_dephasing_rate,time_independent=True)})
        qchannel2 = QuantumChannel("QuantumChannel{}toS".format(qlientname),length=distance, delay=1,
                                   models={"quantum_loss_model": FibreLossModel(p_loss_init=1-fiber_coupling,
                                                                                p_loss_length=fiber_loss),
                                           "quantum_noise_model":DephaseNoiseModel(dephase_rate = fiber_dephasing_rate,time_independent=True)})
        
        
        Qonn_send, Qlient_receive = network.add_connection(
            qonnectorto, qlientname, channel_to=qchannel1, label="quantumS{}".format(qlientname))
        Qlient_send, Qonn_receive = network.add_connection(
            qlientname, qonnectorto, channel_to=qchannel2, label="quantum{}S".format(qlientname))
    
        # Update the Qonnector's properties
        qmem = QuantumProcessor( "QonnectorMemoryTo{}".format(qlientname), num_positions=2 ,
                                phys_instructions=qonnector_physical_instructions)
        Qonnector.add_subcomponent(qmem)
        Qonnector.QlientList.append(qlientname)
        Qonnector.QlientPorts[qlientname] = [Qonn_send,Qonn_receive]
        Qonnector.QlientKeys[qlientname] = []
    
        #Update Qlient ports
        Qlient.listports = [Qlient_send, Qlient_receive]
        
        def route_qubits(msg):
            target = msg.meta.pop('internal', None)

            if isinstance(target, QuantumMemory):
                if not target.has_supercomponent(Qonnector):
                    raise ValueError("Can't internally route to a quantummemory that is not a subcomponent.")
                target.ports['qin'].tx_input(msg)
            else:
                Qonnector.ports[Qonn_send].tx_output(msg)
            
        # Connect the Qonnector's ports
        qmem.ports['qout'].bind_output_handler(route_qubits) #port to send to Qlient
        Qonnector.ports[Qonn_receive].forward_input(qmem.ports["qin"]) #port to receive from Qlient

        # Connect the Qlient's ports 
        Qlient.ports[Qlient_receive].forward_input(Qlient.qmemory.ports["qin"]) #port to receive from qonnector
        Qlient.qmemory.ports["qout"].forward_output(Qlient.ports[Qlient_send]) #port to send to qonnector
        
        #Classical channels on top of that
        cchannel1 = ClassicalChannel("ClassicalChannelSto{}".format(qlientname),length=distance, delay=1)
        cchannel2 = ClassicalChannel("ClassicalChannel{}toS".format(qlientname),length=distance, delay=1)
        
        network.add_connection(qonnectorto, qlientname, channel_to=cchannel1, 
                               label="ClassicalS{}".format(qlientname), port_name_node1="cout_{}".format(qlientname),
                               port_name_node2="cin")
        network.add_connection(qlientname, qonnectorto, channel_to=cchannel2, 
                               label="Classical{}S".format(qlientname), port_name_node1="cout",
                               port_name_node2="cin_{}".format(qlientname))
        
    def Connect_Qonnector(self, Qonnector1, Qonnector2, distmid1,distmid2,tsat1,tsat2, linktype):
    #Method to connect two Qonnectors. It creates dedicated quantum processor at each Qonnector.
    
    ### Parameters ###
    #Qonnector1: name of the first Qonnector (str)
    #Qonnector2: name of the second Qonnector (str)
    #distmid1: distance between Qonnector 1 and middle node (drone or satellite) in km
    #distmid2: distance between Qonnector 2 and middle node (drone or satellite) in km
    #linktype: "drone" or "satellite". In the first case it creates a direct free space link between the two 
    #          Qonnectors. In the "satellite case", it creates a quantum processor (satellite) in the middle 
    #          connected to the two Qonnectors
    
        network = self.network
        
        #check the format of linktype
        if not(linktype == "drone") and not (linktype == "satellite"):
            raise NameError("Wrong link type, it should be drone or satellite")
        
        
        #create dedicated quantum memories at each qonnector
        Qonn1 = network.get_node(Qonnector1)
        Qonn2 = network.get_node(Qonnector2)
        
        #create channels depending on the link type
        if linktype == "drone":
            
            qmem1 = QuantumProcessor( "QonnectorMemoryTo{}".format(Qonnector2), num_positions=2 ,
                                phys_instructions=qonnector_physical_instructions)
            Qonn1.add_subcomponent(qmem1)
            qmem2 = QuantumProcessor( "QonnectorMemoryTo{}".format(Qonnector1), num_positions=2 ,
                                phys_instructions=qonnector_physical_instructions)
            Qonn2.add_subcomponent(qmem2)
            
            qchannel1 = QuantumChannel("FreeSpaceChannelto{}".format(Qonnector2),length=distmid1+distmid2, delay=1,
                                   models={"quantum_loss_model": FreeSpaceLossModel(W0, rx_aperture_freespace,
                                                                                    Cn2_freespace, wavelength, Tatm)})
            
            qchannel2 = QuantumChannel("FreeSpaceChannelto{}".format(Qonnector1),length=distmid1+distmid2, delay=1,
                                   models={"quantum_loss_model": FreeSpaceLossModel(W0, rx_aperture_freespace,
                                                                                    Cn2_freespace, wavelength,Tatm)})
        
        
            #connect the channels to nodes
            Qonn1_send, Qonn2_receive = network.add_connection(
                    Qonn1, Qonn2, channel_to=qchannel1, label="FreeSpaceTo{}".format(Qonnector2))
            Qonn2_send, Qonn1_receive = network.add_connection(
                    Qonn2, Qonn1, channel_to=qchannel2, label="FreeSpaceTo{}".format(Qonnector1))
        
            #update both qonnectors properties
            Qonn1.QlientList.append(Qonnector2)
            Qonn1.QlientPorts[Qonnector2] = [Qonn1_send,Qonn1_receive]
            Qonn1.QlientKeys[Qonnector2] = []
        
            Qonn2.QlientList.append(Qonnector1)
            Qonn2.QlientPorts[Qonnector1] = [Qonn2_send,Qonn2_receive]
            Qonn2.QlientKeys[Qonnector1] = []
        
            # Connect the Qonnector's ports
            def route_qubits1(msg):
                target = msg.meta.pop('internal', None)

                if isinstance(target, QuantumMemory):
                    if not target.has_supercomponent(Qonn1):
                        raise ValueError("Can't internally route to a quantummemory that is not a subcomponent.")
                    target.ports['qin'].tx_input(msg)
                else:
                    Qonn1.ports[Qonn1_send].tx_output(msg)
            
        
            qmem1.ports['qout'].bind_output_handler(route_qubits1) 
            Qonn1.ports[Qonn1_receive].forward_input(qmem1.ports["qin"]) 
        
            def route_qubits2(msg):
                target = msg.meta.pop('internal', None)

                if isinstance(target, QuantumMemory):
                    if not target.has_supercomponent(Qonn2):
                        raise ValueError("Can't internally route to a quantummemory that is not a subcomponent.")
                    target.ports['qin'].tx_input(msg)
                else:
                    Qonn2.ports[Qonn2_send].tx_output(msg)
            
        
            qmem2.ports['qout'].bind_output_handler(route_qubits2) 
            Qonn2.ports[Qonn2_receive].forward_input(qmem2.ports["qin"]) 
        
            #Classical channels on top of that
            cchannel1 = ClassicalChannel("ClassicalChannelto{}".format(Qonnector2),length=distance, delay=1)
            cchannel2 = ClassicalChannel("ClassicalChannelto{}".format(Qonnector1),length=distance, delay=1)
        
            network.add_connection(Qonn1, Qonn2, channel_to=cchannel1, 
                               label="Classicalto{}".format(Qonnector2), port_name_node1="cout_{}".format(Qonnector2),
                               port_name_node2="cin_{}".format(Qonnector1))
            network.add_connection(Qonn2, Qonn1, channel_to=cchannel2, 
                               label="Classicalto{}".format(Qonnector1), port_name_node1="cout_{}".format(Qonnector1),
                               port_name_node2="cin_{}".format(Qonnector2))
            
        if linktype == "satellite":
            
            
            #Create a satellite node with a quantum processor for each qonnector
            Satellite = Qonnector_node("Satellite{}".format(Qonnector1+Qonnector2), QlientList=[],QlientPorts={},QlientKeys={})
            network.add_node(Satellite)
            qmem3 = QuantumProcessor( "SatelliteMemoryTo{}".format(Qonnector1), num_positions=2 ,
                                phys_instructions=qonnector_physical_instructions)
            Satellite.add_subcomponent(qmem3)
            qmem4 = QuantumProcessor( "SatelliteMemoryTo{}".format(Qonnector2), num_positions=2 ,
                                phys_instructions=qonnector_physical_instructions)
            Satellite.add_subcomponent(qmem4)
            
            
            qmem1 = QuantumProcessor( "QonnectorMemoryTo{}".format(Satellite.name), num_positions=2 ,
                                phys_instructions=qonnector_physical_instructions)
            Qonn1.add_subcomponent(qmem1)
            qmem2 = QuantumProcessor( "QonnectorMemoryTo{}".format(Satellite.name), num_positions=2 ,
                                phys_instructions=qonnector_physical_instructions)
            Qonn2.add_subcomponent(qmem2)
            
            
            #Connect Satellite with Qonn1 (only downlink)
            qchannel1 = QuantumChannel("SatChannelto{}".format(Qonnector1),length=distmid1, delay=1,
                                   models={"quantum_loss_model": FixedSatelliteLossModel(txDiv, sigmaPoint,
                                                                            rx_aperture_sat, Cn2_sat, wavelength,tsat1)})
        
            #connect the channels to nodes
            Sat1_send, Qonn1_receive = network.add_connection(
                    Satellite, Qonn1, channel_to=qchannel1, label="SatelliteChanTo{}".format(Qonnector1))

        
            #update both node properties
            Satellite.QlientList.append(Qonnector1)
            Satellite.QlientPorts[Qonnector1] = [Sat1_send]
            Satellite.QlientKeys[Qonnector1] = []
        
            Qonn1.QlientList.append(Satellite.name)
            Qonn1.QlientPorts[Satellite.name] = ['dummyport',Qonn1_receive]
            Qonn1.QlientKeys[Satellite.name] = []
        
            # Connect the Satellite and Qonnector's ports
            def route_qubits3(msg):
                target = msg.meta.pop('internal', None)

                if isinstance(target, QuantumMemory):
                    if not target.has_supercomponent(Satellite):
                        raise ValueError("Can't internally route to a quantummemory that is not a subcomponent.")
                    target.ports['qin'].tx_input(msg)
                else:
                    Satellite.ports[Sat1_send].tx_output(msg)
            
        
            qmem3.ports['qout'].bind_output_handler(route_qubits3) 
        
             
            Qonn1.ports[Qonn1_receive].forward_input(qmem1.ports["qin"]) 
        
            #Classical channels on top of that
            cchannel1 = ClassicalChannel("ClassicalChannelto{}".format(Qonnector1),length=distmid1, delay=1)
            cchannel2 = ClassicalChannel("ClassicalChanneltoSatellite",length=distmid1, delay=1)
        
            network.add_connection(Satellite, Qonn1, channel_to=cchannel1, 
                               label="Classicalto{}".format(Qonnector1), port_name_node1="cout_{}".format(Qonnector1),
                               port_name_node2="cin_{}".format(Satellite.name))
            network.add_connection(Qonn1, Satellite, channel_to=cchannel2, 
                               label="ClassicaltoSat".format(Qonnector1), port_name_node1="cout_{}".format(Satellite.name),
                               port_name_node2="cin_{}".format(Qonnector1))
            
            #Do the same with Qonn2
            qchannel2 = QuantumChannel("SatChannelto{}".format(Qonnector2),length=distmid2, delay=1,
                                   models={"quantum_loss_model": FixedSatelliteLossModel(txDiv, sigmaPoint,
                                                                            rx_aperture_sat, Cn2_sat, wavelength,tsat2)})
        
            #connect the channels to nodes
            Sat2_send, Qonn2_receive = network.add_connection(
                    Satellite, Qonn2, channel_to=qchannel2, label="SatelliteChanTo{}".format(Qonnector2))

        
            #update both node properties
            Satellite.QlientList.append(Qonnector2)
            Satellite.QlientPorts[Qonnector2] = [Sat2_send]
            Satellite.QlientKeys[Qonnector2] = []
        
            Qonn2.QlientList.append(Satellite.name)
            Qonn2.QlientPorts[Satellite.name] = ['dummyport',Qonn2_receive]
            Qonn2.QlientKeys[Satellite.name] = []
        
            # Connect the Satellite and Qonnector's ports
            def route_qubits4(msg):
                target = msg.meta.pop('internal', None)

                if isinstance(target, QuantumMemory):
                    if not target.has_supercomponent(Satellite):
                        raise ValueError("Can't internally route to a quantummemory that is not a subcomponent.")
                    target.ports['qin'].tx_input(msg)
                else:
                    Satellite.ports[Sat2_send].tx_output(msg)
            
        
            qmem4.ports['qout'].bind_output_handler(route_qubits4) 
        
             
            Qonn2.ports[Qonn2_receive].forward_input(qmem2.ports["qin"]) 
        
            #Classical channels on top of that
            cchannel3 = ClassicalChannel("ClassicalChannelto{}".format(Qonnector2),length=distmid2, delay=1)
            cchannel4 = ClassicalChannel("ClassicalChanneltoSatellite",length=distmid2, delay=1)
        
            network.add_connection(Satellite, Qonn2, channel_to=cchannel3, 
                               label="Classicalto{}".format(Qonnector2), port_name_node1="cout_{}".format(Qonnector2),
                               port_name_node2="cin_{}".format(Satellite.name))
            network.add_connection(Qonn2, Satellite, channel_to=cchannel4, 
                               label="ClassicaltoSat".format(Qonnector2), port_name_node1="cout_{}".format(Satellite.name),
                               port_name_node2="cin_{}".format(Qonnector2))


class SendBB84(NodeProtocol):
    
    #Protocol performed by a node to send a random BB84 qubit. It assumes the network is already created and 
    # the node are well connected
    
    #Parameter:
    # node: sending node
    # othernode: receiving node
    # init_succ: probability that a qubit creation attempt succeeds
    
    def __init__(self,othernode, init_succ, init_flip,node):
        super().__init__(node=node)
        self._othernode = othernode
        self._init_succ = init_succ
        self._init_flip = init_flip
    
    def run(self):
        if self.node.name[0:9] == 'Qonnector' or self.node.name[0:9]== 'Satellite':
            
            if self.node.name[0:9]== 'Satellite':
                mem = self.node.subcomponents["SatelliteMemoryTo{}".format(self._othernode.name)]
            else:
                mem = self.node.subcomponents["QonnectorMemoryTo{}".format(self._othernode.name)]
        
            
            clock = Clock(name="clock",
                      start_delay=0,
                      models={"timing_model": FixedDelayModel(delay=Qonnector_init_time )})
            self.node.add_subcomponent(clock)
            clock.start()
        
            while True:
                mem.reset()

                mem.execute_instruction(instr.INSTR_INIT,[0])
                yield self.await_program(mem,await_done=True,await_fail=True)
                #print("qubit created")
                succ = bernoulli.rvs(self._init_succ)
                if (succ == 1):                    
                    flip = bernoulli.rvs(self._init_flip)
                    if (flip == 1):
                        mem.execute_instruction(instr.INSTR_X, [0], physical = False)
            
                    base = bernoulli.rvs(0.5) #random choice of a basis
                    if base <0.5:
                        mem.execute_instruction(instr.INSTR_H,[0])
                        base = "plusmoins"
                    else:
                        mem.execute_instruction(instr.INSTR_I,[0])
                        base = "zeroun"
                
                    yield self.await_program(mem,await_done=True,await_fail=True)
                
                    t = clock.num_ticks
                    bit = bernoulli.rvs(0.5) #random choice of a bit
                    if bit < 0.5:
                        mem.execute_instruction(instr.INSTR_I, [0], physical=False)
                        self.node.QlientKeys[self._othernode.name].append(([t,base],0))
                    else:
                        if base == "zeroun":
                            mem.execute_instruction(instr.INSTR_X, [0], physical=False)
                        elif base == "plusmoins":
                            mem.execute_instruction(instr.INSTR_Z, [0], physical=False)
                        self.node.QlientKeys[self._othernode.name].append(([t,base],1))
                
                    qubit, = mem.pop([0])
                    self.node.ports["cout_{}".format(self._othernode.name)].tx_output(t)
                
                
        else:
            mem = self.node.qmemory
            clock = Clock(name="clock",start_delay=0,
                      models={"timing_model": FixedDelayModel(delay=Qlient_init_time )})
            
            self.node.add_subcomponent(clock)
            clock.start()
            
            while True:
                mem.reset()

                mem.execute_instruction(instr.INSTR_INIT,[0])
                yield self.await_program(mem,await_done=True,await_fail=True)
                    #print("qubit created")
                succ = bernoulli.rvs(self._init_succ)
                if (succ == 1):      
                    flip = bernoulli.rvs(self._init_flip)
                    if (flip == 1):
                        mem.execute_instruction(instr.INSTR_X, [0], physical = False)
            
                    base = bernoulli.rvs(0.5) #random choice of a basis
                    if base <0.5:
                        mem.execute_instruction(instr.INSTR_H,[0])
                        base = "plusmoins"
                    else:
                        mem.execute_instruction(instr.INSTR_I,[0])
                        base = "zeroun"
            
                    yield self.await_program(mem,await_done=True,await_fail=True)
                
                    t = clock.num_ticks
                    bit = bernoulli.rvs(0.5) #random choice of a bit
                    if bit < 0.5:
                        mem.execute_instruction(instr.INSTR_I, [0], physical=False)
                        self.node.keylist.append(([t,base],0))
                    else:
                        if base == "zeroun":
                            mem.execute_instruction(instr.INSTR_X, [0], physical=False)
                        elif base == "plusmoins":
                            mem.execute_instruction(instr.INSTR_Z, [0], physical=False)
                        self.node.keylist.append(([t,base],1))
            
                    qubit, = mem.pop([0])
                    self.node.ports["cout"].tx_output(t)
                
                    #print("qubit sent")
            

class ReceiveProtocol(NodeProtocol):
    
    # Protocol performed by a node to receive a state a measure it
    
    #Parameters:
    # othernode: node from which a qubit is expected
    # measurement_succ: probability that the measurement succeeds
    # measurement_flip: probability that the detector flips the outcome (crosstalk)
    # BB84: boolean indicating if we perform BB84 measurement (random choice of measurement basis)
    
        def __init__(self, othernode, measurement_succ, measurement_flip, BB84, node):
            super().__init__(node=node)
            self._othernode = othernode
            self._measurement_succ=measurement_succ
            self._BB84 = BB84
            self._measurement_flip = measurement_flip

        def run(self):
            if self.node.name[0:9] == 'Qonnector':
                mem = self.node.subcomponents["QonnectorMemoryTo{}".format(self._othernode.name)]
                port = self.node.ports[self.node.QlientPorts[self._othernode.name][1]]
                #print(port)
                while True:
                    yield self.await_port_input(port)
                    t = self.node.ports["cin_{}".format(self._othernode.name)].rx_input()
                    
                    b = bernoulli.rvs(self._measurement_succ)
                    
                    if b ==1 :
                        if self._BB84: #in case we perform BB84
                            base = bernoulli.rvs(0.5) #choose a random basis
                            if base < 0.5:
                                mem.execute_instruction(instr.INSTR_H, [0], physical = False)
                                base = "plusmoins"
                            else:
                                mem.execute_instruction(instr.INSTR_I, [0],physical = False)
                                base = "zeroun"
                        else:
                            base = None 
                        
                        m,_,_ = mem.execute_instruction(instr.INSTR_MEASURE,[0],output_key="M1")
                        yield self.await_program(mem,await_done=True,await_fail=True)
                        
                        flip = bernoulli.rvs(self._measurement_flip)
                        if (flip==1):
                            if m['M1'][0]==0:
                                m['M1'][0] =1
                            elif m['M1'][0]==1:
                                m['M1'][0]=0
                            
                        if m['M1'] is not None and t is not None and base is not None:
                            self.node.QlientKeys[self._othernode.name].append(([t.items[0],base],m['M1'][0]))
                            
                        elif m['M1'] is not None and t is not None:
                            self.node.QlientKeys[self._othernode.name].append((t.items,m['M1'][0]))
                            
                        elif m['M1'] is not None:
                            self.node.QlientKeys[self._othernode.name].append(m['M1'][0])
                    mem.reset()
                        
            
            else:
                mem = self.node.qmemory
                port = self.node.ports[self.node.listports[1]]
                
                while True:
                    yield self.await_port_input(port)
                    #print("qubit received")
                    #qubit, = mem.peek([0])
                    #print(mem.peek([0]))
                    t = self.node.ports["cin"].rx_input()
                    
                    b = bernoulli.rvs(self._measurement_succ)
                    #print(b)
                    if b ==1 :
                        if self._BB84: #in case we perform BB84
                            base = bernoulli.rvs(0.5) #choose a random basis
                            if base < 0.5:
                                mem.execute_instruction(instr.INSTR_H, [0], physical = False)
                                base = "plusmoins"
                            else:
                                mem.execute_instruction(instr.INSTR_I, [0],physical = False)
                                base = "zeroun"
                        else:
                            base = None 
                            
                        if not(mem.busy):
                            
                            
                            m,_,_ = mem.execute_instruction(instr.INSTR_MEASURE,[0],output_key="M1")
                            yield self.await_program(mem,await_done=True,await_fail=True)
                            #print("qubit measured")
                            
                            flip = bernoulli.rvs(self._measurement_flip)
                            if (flip==1):
                                if m['M1'][0]==0:
                                    m['M1'][0] =1
                                elif m['M1'][0]==1:
                                    m['M1'][0]=0
                                    
                            if m['M1'] is not None and t is not None and base is not None:
                                self.node.keylist.append(([t.items[0], base],m['M1'][0]))
                            
                            elif m['M1'] is not None and t is not None:
                                self.node.keylist.append((t.items,m['M1'][0]))
                        
                            elif m['M1'] is not None:
                                self.node.keylist.append(m['M1'][0])
                            
                    mem.reset()
                
                
                
class  TransmitProtocol(NodeProtocol):
    #Protocol performed by a Qonnector to transmit a qubit sent by a Qlient to another Qlient
        
        #Parameters
        # Qlient_from: node from which a qubit is expected
        # Qlient_to: node to which transmit the qubit received
        # switch_succ: probability that the transmission succeeds
        
        def __init__(self, Qlient_from, Qlient_to, switch_succ, node=None, name=None):
                super().__init__(node=node, name=name)
                self._Qlient_from = Qlient_from
                self._Qlient_to = Qlient_to
                self._switch_succ=switch_succ
        
        def run(self):
            rec_mem = self.node.subcomponents["QonnectorMemoryTo{}".format(self._Qlient_from.name)]
            rec_port = self.node.ports[self.node.QlientPorts[self._Qlient_from.name][1]]
            sen_mem = self.node.subcomponents["QonnectorMemoryTo{}".format(self._Qlient_to.name)]
            
            while True:
                rec_mem.reset()
                sen_mem.reset()
                
                yield self.await_port_input(rec_port)
                #print("qubit received at qonnector" )
                t = self.node.ports["cin_{}".format(self._Qlient_from.name)].rx_input()
                
                rec_mem.pop([0], skip_noise=True, meta_data={'internal': sen_mem})
                #print("qubit moved in qonnector's memory")
                
                
                b = bernoulli.rvs(self._switch_succ)
                if b ==1 :
                    qubit, = sen_mem.pop([0])
                    self.node.ports["cout_{}".format(self._Qlient_to.name)].tx_output(t)
                    #print("qubit sent to node")
                    
                    
class SendEPR(NodeProtocol):
    #Protocol performed by a Qonnector node to create a send EPR pairs to two Qlient, each getting one qubit
    
    def __init__(self, Qlient_1, Qlient_2, EPR_succ, node = None, name = None):
        super().__init__(node=node, name=name)
        self._Qlient_1 = Qlient_1
        self._Qlient_2 = Qlient_2
        self._EPR_succ = EPR_succ
        
    def run(self):
        if self.node.name[0:9]== 'Satellite':
            mem1 = self.node.subcomponents["SatelliteMemoryTo{}".format(self._Qlient_1.name)]
            mem2 = self.node.subcomponents["SatelliteMemoryTo{}".format(self._Qlient_2.name)]
            port1 = self.node.ports[self.node.QlientPorts[self._Qlient_1.name][0]]
        else:
            mem1 = self.node.subcomponents["QonnectorMemoryTo{}".format(self._Qlient_1.name)]
            mem2 = self.node.subcomponents["QonnectorMemoryTo{}".format(self._Qlient_2.name)]
            port1 = self.node.ports[self.node.QlientPorts[self._Qlient_1.name][1]]
        state_sampler = StateSampler(qreprs=[ks.b11],
                                 probabilities=[1])

        qsource = QSource("qsource{}".format(self._Qlient_1.name+self._Qlient_2.name),
                          state_sampler=state_sampler,
                          num_ports=2,
                          timing_model=FixedDelayModel(delay=EPR_time),
                          status=SourceStatus.EXTERNAL)
        clock = Clock(name="clock",
                      start_delay=0,
                      models={"timing_model": FixedDelayModel(delay=EPR_time)})
        
        self.node.add_subcomponent(clock)
        self.node.add_subcomponent(qsource)
        clock.ports["cout"].connect(qsource.ports["trigger"])
        
        qsource.ports["qout0"].connect(mem1.ports["qin"])
        qsource.ports["qout1"].connect(mem2.ports["qin"])
        clock.start()
        
        while True:
            yield self.await_port_input(mem1.ports["qin"]) and self.await_port_input(mem2.ports["qin"])
            
            b = bernoulli.rvs(self._EPR_succ)
            if b==1:
                mem1.pop([0])
                self.node.ports["cout_{}".format(self._Qlient_1.name)].tx_output(clock.num_ticks)
                self.node.QlientKeys[self._Qlient_1.name].append(0)
                mem2.pop([0])
                self.node.ports["cout_{}".format(self._Qlient_2.name)].tx_output(clock.num_ticks)
                self.node.QlientKeys[self._Qlient_2.name].append(0)
                
            mem1.reset()
            mem2.reset()
            


def Sifting(Lalice, Lbob):
    #Function to get the number of matching received qubit. If BB84 then the resulting list contains the qubits
    # that were sent and measured in the same basis. If EPR then the resulting list contains the qubit measured
    # by Alice and Bob that came from the same EPR pair
    Lres = []
    for i in range(len(Lalice)):
        ta, ma = Lalice[i]
        for j in range(len(Lbob)):
            tb, mb = Lbob[j]
            if ta == tb:
                Lres.append((ma,mb))
        
    return Lres

