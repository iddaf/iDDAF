# def __init__(self): 
#     import numpy as np
    
def Calc_Percent_of_Noise (Z_est, Z_mat, Threshold, fullRank, Verbose):
    
    import numpy as np
    
                  
    if fullRank != True:
#         if Verbose == "True": 
#             print("System Unobservable!"
        print("System Unobservable!\n Returning Null")
        return None, None, Z_mat[:,0]        
        
    
    Z_mat = Z_mat[Z_mat[:,0].argsort(kind='mergesort')]

    # Calculating the measurement error
    M_Noise_copy = (Z_mat[:,2] - Z_est)
    M_Noise = np.absolute(M_Noise_copy).copy()
    
    # putting error for not measured measurements as 0
    M_Noise[0] = 0
    M_Noise[Z_mat[:,1] == 0] = 0
    M_Noise[Z_mat[:,1] == -1] = 0
    
    for index in range (M_Noise.size):
        if np.absolute(Z_mat[index,2])<0.1 and np.absolute(M_Noise[index])<0.1:
            M_Noise[index] = 0
        
    #############################################################
    # Calcualating the perecent of noise
    non0indx = np.where(Z_mat[:,2]!=0)
    P_Noise=np.zeros(M_Noise.shape)
    P_Noise[non0indx] = np.absolute(M_Noise[non0indx] / Z_mat[non0indx,2] * 100)
    Noisy_index = np.where(P_Noise > Threshold)[0]

    if Verbose == "True":
        print("\nM_Noise: \n", M_Noise_copy)
        print("\nP_Noise: \n", P_Noise)
        print("\n Noisy_index: \n")
        print(Noisy_index)

    return M_Noise_copy, P_Noise, Noisy_index


def SE_BDD(H_mat, Z_mat, Threshold, Verbose):
    print("Threshold: ", Threshold)
    import numpy as np
    from numpy.random import seed
    from numpy.random import randn
    from numpy import mean
    from numpy import std
    
    Threshold_max = 10
    ThresholdBDD = 2
    
    Z_mat = Z_mat[Z_mat[:,0].argsort(kind='mergesort')]
    Z_mat_Copy = Z_mat.copy()
    finalCheck = 1
    
    #print("\n **************  State Estimation ************* \n")
    #takes H matrix Z measurements, Threshold and number of Equations as input
    #print("Considering only the taken measurements")
    Z_msr_init = Z_mat[Z_mat[:,1]==1][:,2]
    
    # considering only the corresponding columns in H   
    H_mat_init = H_mat[Z_mat[:,1]==1]

    #printing the sizes of H, Rank (H) and Z
    Rank=np.linalg.matrix_rank(H_mat_init)
    #print("Rank" , Rank)
    Z_est = np.zeros(H_mat.shape[0])
    
    if Rank == H_mat.shape[1]:
        
        fullRank = True
        # Estimating the states using WMSE estimator
        #States_init = np.linalg.pinv(H_mat_init)@Z_msr_init
        States_init = (np.linalg.inv(H_mat_init.T@H_mat_init)@H_mat_init.T)@Z_msr_init

        # Estimating the measurement from the estimated states
        Z_est = H_mat@States_init

        if Verbose == "True":
            print("\n Initial H: "+str(H_mat_init.shape))
            print("\n Rank H: "+str(Rank))
            print("\n Initital Z_m: "+str(Z_msr_init.shape))
            print("\n Sates: \n"+str(States_init))
            print("\n Z_est: \n"+str(Z_est))


        # Calculating the Noise
        [M_Noise, P_Noise, Noisy_index] = Calc_Percent_of_Noise (Z_est.copy(), Z_mat.copy(), ThresholdBDD, fullRank,  Verbose)

        if(len(Noisy_index) > 0): 

            #print("***************** Bad Data Detected *********************")
            #changing the noisy measurement parameter (taken) from 1 to 0
            
            data_mean, data_std = mean(P_Noise), std(P_Noise)
            
            #print(data_mean, data_std)
            # identify outliers
            cut_off = data_std * 1.2
            upper = data_mean + cut_off
            indxtbremoved = np.where(P_Noise > upper)[0]
            
            if indxtbremoved.size > 0:
                temp_indx = indxtbremoved
                
            else:
                noisyIdxsorted = np.argsort(P_Noise)[::-1]
                #temp_indx = noisyIdxsorted[0]
                temp_indx = noisyIdxsorted
                
            if Verbose == "True":                       
                print("Removing sensor: ", temp_indx)
            
            #############################################################
            temp=Z_mat[:,1].copy()
            temp[temp_indx]= -1
            Z_mat[:,1]=temp.copy()
            

            # considering only the corresponding columns in H   
            #printing the sizes of H, Rank (H) and Z
            H_mat_check = H_mat[Z_mat[:,1]==1]
            
            if H_mat_check.shape[0] > 0:
                RankCheck=np.linalg.matrix_rank(H_mat_check)
                
            else: RankCheck = -1 
            
            
            if Verbose == "True":
                print("\nModified Z: \n"+str(Z_mat))
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                
            if RankCheck == H_mat.shape[1]:
                
                fullRank = True
                [States_init, Z_est, Z_mat, fullRank, P_Noise]= SE_BDD (H_mat, Z_mat, Threshold, Verbose)
            
            else: 
                print("Cant remove this noisy measurement! Singularity warning!!")
                #fullRank = False
                
                if Threshold < Threshold_max:
                    Threshold = 2 + Threshold
                    #Threshold = 2* Threshold
                    fullRank =True
                    print("Increasing Threshold..:", Threshold)
                    #print("fullRank: ", fullRank)
                    Z_mat[np.where(Z_mat[:, 1] == -1)] = 1 
                    [States_init, Z_est, Z_mat, fullRank, P_Noise]= SE_BDD (H_mat, Z_mat, Threshold, Verbose)
                else:
                    fullRank = False
                    Rank = RankCheck
                #States_init = 0
                #Z_est = 0
                #P_Noise = 0
                #return States_init, Z_est, Z_mat, fullRank, P_Noise
            #print("***************** End of SE **********************")
    else:
        print("The H is not a full rank matrix !! second if")

        #if Verbose == "Final":
        fullRank = False
        ####################
        
        if Threshold < Threshold_max:
            Threshold = 2 + Threshold
            #Threshold = 2* Threshold
            fullRank =True
            print("Increasing Threshold..:", Threshold)
            #print("fullRank: ", fullRank)
            Z_mat[np.where(Z_mat[:, 1] == -1)] = 1 
            [States_init, Z_est, Z_mat, fullRank, P_Noise]= SE_BDD (H_mat, Z_mat, Threshold, Verbose)
#################################
        else:
            States_init = 0
            Z_est = 0
            #Z_mat = 0
            P_Noise = 0
            return States_init, Z_est, Z_mat, fullRank, P_Noise

    if Verbose != "Final" :
        print("Final Checcking... with Threshold:", Threshold)
        Verbose = "Final"
        Z_mat = Z_mat_Copy.copy()
        
        if fullRank == True:
            print("fullRank == True")
            [M_Noise, P_Noise, Noisy_index] = Calc_Percent_of_Noise (Z_est.copy(), Z_mat.copy(), ThresholdBDD, fullRank, Verbose)
        #print("Noisy_index", Noisy_index)
        #print("Z_mat: ", Z_mat)
            if Noisy_index.size > 0 and fullRank == True: 
                Z_mat[Noisy_index, 1] = -1 
                print("Calling State Estimation again... in the final steps")
                [States_init, Z_est, Z_mat, fullRank, P_Noise] = SE_BDD (H_mat.copy(), Z_mat.copy(), Threshold, Verbose)
        
#         if Noisy_index.size>0 and fullRank == True: 
#             Z_mat[Noisy_index, 1] = -1 
#         [States_init, Z_est, Z_mat, fullRank, P_Noise] = SE_BDD (H_mat.copy(), Z_mat.copy(), Threshold, Verbose)
#         ######################################
        
    return States_init, Z_est, Z_mat, fullRank, P_Noise

def SE_BDD_Noise(H_mat, Z_mat, Threshold, Verbose):
    
    import numpy as np
    from numpy.random import seed
    from numpy.random import randn
    from numpy import mean
    from numpy import std
    
    Z_mat = Z_mat[Z_mat[:,0].argsort(kind='mergesort')]
    Z_mat_Copy = Z_mat.copy()
    
    #print("\n **************  State Estimation ************* \n")
    #takes H matrix Z measurements, Threshold and number of Equations as input
    #print("Considering only the taken measurements")
    
    Z_msr_init = Z_mat[Z_mat[:,1]==1][:,2]
    
    # considering only the corresponding columns in H   
    
    H_mat_init = H_mat[Z_mat[:,1]==1]

    #printing the sizes of H, Rank (H) and Z
    
    Rank = np.linalg.matrix_rank(H_mat_init)
    print("Rank of Noisy: " , Rank)
    
    Z_est = np.zeros(H_mat.shape[0])
    
    if Rank == H_mat.shape[1]:
        
        fullRank = True
        # Estimating the states using WMSE estimator
        #States_init = np.linalg.pinv(H_mat_init)@Z_msr_init
        States_init = (np.linalg.inv(H_mat_init.T@H_mat_init)@H_mat_init.T)@Z_msr_init

        # Estimating the measurement from the estimated states
        Z_est = H_mat@States_init
        ################################
        print("SE_BDD_Noise - Z_est:\n", Z_est)
        print("Mag: ", np.linalg.norm(Z_est))
        ################################
        [M_Noise, P_Noise, Noisy_index] = Calc_Percent_of_Noise (
            Z_est.copy(), Z_mat.copy(), Threshold, fullRank, Verbose)
        ##########################################################
        if Verbose == "True":
            print("\n Initial H: "+str(H_mat_init.shape))
            print("\n Rank H: "+str(Rank))
            print("\n Initital Z_m: "+str(Z_msr_init.shape))
            print("\n Sates: \n"+str(States_init))
            print("\n Z_est: \n"+str(Z_est))
    else:
        
        #print("The H is not a full rank matrix !!")
        fullRank = False
        States_init = 0
        #Z_est = 0
        P_Noise = 0
        
    return States_init, Z_est, Z_mat, fullRank, P_Noise
###############################################
def randID (Z, NumOfDeception):
    import numpy as np
    import matplotlib.pyplot as plt
    #print(Z)
    senID = Z[1:,0].astype(int)
    
    np.random.shuffle(senID)
    selectedIDs = senID[0:NumOfDeception].copy()
    selectedIDs = selectedIDs[np.argsort(selectedIDs)]
    #print(selectedIDs)
    deceptiveIDs = selectedIDs.copy() 
    #print("deceptiveIDs earlier: ", deceptiveIDs)
    count = 0
    while(np.where(selectedIDs == deceptiveIDs)[0].size != 0):
        count = count + 1

        #print("Shuffling")
        #print("deceptiveIDs after: ", deceptiveIDs)

        np.random.shuffle(deceptiveIDs)
        if count > 100:
            break
    #print("deceptiveIDs after: ", deceptiveIDs)
        
    Z[selectedIDs,0] = deceptiveIDs
    #print(Z)
    Z = Z[Z[:,0].argsort(kind='mergesort')]
    return Z, selectedIDs, deceptiveIDs

def fixID(Z, selectedIDs, deceptiveIDs):
    Z[deceptiveIDs,0] = selectedIDs.copy()
    Z = Z[Z[:,0].argsort(kind='mergesort')]
    return Z

def shuffleID(Z, selectedIDs, deceptiveIDs):
    Z[selectedIDs,0] = deceptiveIDs.copy()
    Z = Z[Z[:,0].argsort(kind='mergesort')]
    return Z

def mapOrgID(attackedIndx, selectedIDs, deceptiveIDs):
    import numpy as np
    IDs =[]
    for i in attackedIndx:
        found = np.where(deceptiveIDs==i)[0]
        if found.size > 0:          
            IDs.append(selectedIDs[found[0]])
    IDs = np.array(IDs)
    return IDs

####################################################
def findNeighbor (Topo, busID):
    neighborList = []
    for lines in Topo:
        if lines[1] == busID:
            neighborList.append(lines[2])
        elif lines[2] == busID:
            neighborList.append(lines[1])
    neighborList.append(busID)
    return np.array(neighborList)
     

def assignClusterID (sensorList, Topo, NumOfBuses, NumOfLines):
    import numpy as np
    clusterList = []
    clusterPopu = np.zeros(NumOfBuses, dtype = int)
    for sensorID in sensorList:
        if sensorID > 2 * NumOfLines:
            clusterID = sensorID - 2 * NumOfLines
        elif sensorID > NumOfLines:
            lineID = sensorID - NumOfLines
            clusterID = Topo[lineID-1, 2]
        elif sensorID > 0:
            clusterID = Topo[sensorID-1, 1]        
        clusterList.append(clusterID)
        clusterPopu [clusterID-1] += 1
    clusterID = np.array(clusterList)
    return clusterID, clusterPopu  

def selectCluster(promisingClusters, population, clusterPopupulation):
    import numpy as np

    clusterPop = clusterPopupulation[promisingClusters - 1]
    promisingClusters_sorted = promisingClusters[np.argsort(clusterPop)][::-1]
#     indxx= np.arange(0, promisingClusters.size)
#     np.random.shuffle(indxx)
#     return promisingClusters[indxx[0:population]]
    selectedClusters =  promisingClusters_sorted[0:population]
    clusterPopupulation[selectedClusters -1] = clusterPopupulation[selectedClusters - 1] -1
    return selectedClusters

def selectID(selectedClusters, clusterID, selectedCheck, selectedIDs_List):
    import numpy as np

#     print("************  Selecting IDS  ******************")
#     print("selectedClusters: ", selectedClusters)
#     print("selectedCheck :", selectedCheck)
#     print("clusterID :", clusterID)
    
    selectedSensor= []
    
    for cluster in selectedClusters:
        #print("Cluster: ", cluster)
        
        Sensors = np.where(clusterID == cluster)[0]
        #print("Sensors in the cluster: ", Sensors +1)
        #print(Sensors)
        Sensors = Sensors[np.where(selectedCheck[Sensors] == 0)[0]] + 1
        
        #print("Sensors still available : ", Sensors)
        np.random.shuffle(Sensors)
        
        #print("Selected Sensor :", Sensors[0])
        
        selectedSensor.append(Sensors[0])
        
        selectedCheck[Sensors[0]-1] = 1
    #print("#################################")
    #print(" selectedSensors :", selectedSensor)
    #print("selectedCheck :", selectedCheck)
    selectedIDs_List.append(selectedSensor)
    
def doAND(bolArray):
    import numpy as np
    if len(bolArray) == 1:
        return bolArray[0]
    return np.logical_and(bolArray[0], doAND(bolArray[1:]))

def shuffleCluster(cluster, shuffleFlag):
    import numpy as np
    
    sensorList = []
    
    for eachTuple in cluster:
        sensorList.append(eachTuple[0])
    sensorList_org = np.array(sensorList).copy()
    sensorList_dec = sensorList_org.copy()
    
    # deceiving at least one measurement
    if np.count_nonzero(shuffleFlag == 1) > 1:
        cropedSensorList_org = sensorList_org[np.where(shuffleFlag == 1)]
        cropedSensorList_dec = cropedSensorList_org.copy()

        while(doAND(cropedSensorList_dec != cropedSensorList_org) == False):
            np.random.shuffle(cropedSensorList_dec)
            #print("randomizing")

        sensorList_dec[np.where(shuffleFlag == 1)] = cropedSensorList_dec

    new_cluster = []
    
    for eachTuple, sensorID_dec in zip(cluster, sensorList_dec) :
        new_cluster.append((sensorID_dec, eachTuple[1]))
#     print(sensorList_org)
#     print(sensorList_dec)
    return (new_cluster, sensorList_org, sensorList_dec, shuffleFlag)


def process_data (layerList, sublayerList_Dec, shuffleFlag, sensorData_List):
    import numpy as np
    #####  hub data processor ####
    identifier = 1
    for layer in layerList:
        #print(layer)
        #shuffleFlagHub = []
        #print("Collecting layer data: ", identifier, layer)
        for sublayerID in layer:
            #print("sublayer ID: ", sublayerID)

            sublayerData = sublayerList_Dec[sublayerID-1][0]
            sublayer_shuffle_Flags = sublayerList_Dec[sublayerID-1][3]

            for sensorData, flag in zip(sublayerData, sublayer_shuffle_Flags):
                shuffleFlag[identifier-1].append(flag) 
                sensorData_List[identifier-1].append(sensorData)
        identifier += 1
    #####################################################
    #print(hubSensorData_List, shuffleFlagHub)
    return sensorData_List, shuffleFlag

def randomize(sensorData_List, shuffleFlag):
    import numpy as np
    ##### randomizing in hub level ####### 
    sensorData_Dec = []
    #identifier = 1
    for data_, shuffleFlag_ in zip(sensorData_List, shuffleFlag):
        #print("Decepting hub: ", identifier)
        data = shuffleCluster (data_.copy(), np.array(shuffleFlag_).copy())
        sensorData_Dec.append(data)
        #identifier += 1
############################################################
    #print(sensorData_Dec)
    return sensorData_Dec

def create_Zmat(hubList_Dec, Z_mat):
    import numpy as np
    ###########################################
    Z_data_hub = []
    sensorID_org_hub = []
    sensorID_dec_hub = []

    for hubDec in hubList_Dec:
        for sensorData in hubDec[0]:
            Z_data_hub.append(sensorData)

        for orgIDList in hubDec[1]:     
            sensorID_org_hub.append(orgIDList)

        for decIDList in hubDec[2]:
            sensorID_dec_hub.append(decIDList)

    Z_data_hub = np.array(Z_data_hub)
    sensorID_org_hub = np.array(sensorID_org_hub)
    sensorID_dec_hub = np.array(sensorID_dec_hub)
    

    ###############################
    Z_hub = Z_mat.copy()
    Z_hub[1:,0] =  Z_data_hub[:, 0]
    Z_hub[1:,2] =  Z_data_hub[:, 1]
    Z_hub = Z_hub [Z_hub[:,0].argsort(kind = 'mergesort')]

    return Z_hub, sensorID_org_hub, sensorID_dec_hub

def Zmat_at_EMS (Z_mat, NumOfZ):
    import numpy as np
    Z_ems = np.zeros((NumOfZ+1,3), dtype = float)
    Z_ems[Z_mat[:,0].astype(int)] = Z_mat
    Z_ems[:, 0] = np.arange(0,NumOfZ+1)
    return Z_ems



# distribute the sensors over different sections
def id_Dist(n, r):
    if r == 0:
        print(" number of sections cant be zero")
        return
    list_ = []
    avg_frac = n/r
    avg_int = n//r
    
    diff = (avg_frac - avg_int)
    
    if diff >= 0.5:
        diff = 1 - diff
        step = avg_int + 1
        ceil_sign = -1
    else:
        step = avg_int
        ceil_sign = +1
    
    liability = 0
    repeat = 0
    
    while repeat < r:
        repeat = repeat + 1   
        liability = liability + diff * ceil_sign
    
        if abs(liability) >= 0.99:
            added = round(step + liability)
            list_.append(added)
            n = n - added
            liability = liability - ceil_sign

        else:
            list_.append(round(step))
            n = n - round(step)
    return list_

# print(id_Dist(31, 32)) 

###############################
def readX (xList, xSensorData_List, shuffleFlagx, netdata, lineID):
    # rading input
    
    while(True):
        # reading the line
        if lineID >= len(netdata):
            print("Missing data")
            return
        line = netdata[lineID][0]
        lineID = lineID + 1

        #checking for # or spaces
        if line[0] == '#' or line[0] == '\n':
            continue

        # found values
        else: 
            numOfx = int (line.strip())
            #print("numOfx:", numOfx)

            # hub blank list appending
            for index in range (numOfx):
                xSensorData_List.append([])
                shuffleFlagx.append([])

            # checking for data
            index = 0
            while(index < numOfx):

                # reading the line
                if lineID >= len(netdata):
                    print("Missing data")
                    return
                line = netdata[lineID]
                lineID = lineID + 1
                #print("\n\nline: ", line, "index: ", index)


                #checking for # or spaces
                if line[0] == '#' or line[0] == '\n':
                    #print("Passing")
                    continue

                # found values
                else: 
                    #print("Updating....")
                    xList.append([int(i) for i in line.strip().split(", ")]) 
                    index = index +1
                    #print("index: ", index)
        #print(xList)
        break
        
    return xList, xSensorData_List, shuffleFlagx, numOfx, lineID
  #########################################################
    
def pattern_gen(n, capacity):
    import numpy as np

    r = len(capacity)
    if r == 0:
        print(" number of sections cant be zero")
        return
    list_ = []
    avg_frac = n/r
    avg_int = n//r
    
    diff = (avg_frac - avg_int)
    
    if diff >= 0.5:
        diff = 1 - diff
        step = avg_int + 1
        ceil_sign = -1
    else:
        step = avg_int
        ceil_sign = +1
    
    liability = 0
    repeat = 0
    
    while repeat < r:
        repeat = repeat + 1   
        liability = liability + diff * ceil_sign
    
        if abs(liability) >= 0.99:
            added = round(step + liability)
            list_.append(added)
            n = n - added
            liability = liability - ceil_sign

        else:
            list_.append(round(step))
            n = n - round(step)
    ##################################
    #########  checking capacity ##3##
    #print(list_)
    cap = np.array(capacity)
    dis = np.array(list_)
    diff = cap - dis
    #print("diff:", diff)
    
    lagg = np.sum(diff[np.where(diff<0)])
    #print("lagg:", lagg)
    index = 0
    while(lagg < 0):
        positiveIndex = np.where(diff > 0)[0]
        if index >= positiveIndex.size:
            index = 0
        if positiveIndex.size == 0:
            print("Distribution not possible!!!!!")
            break
#         print(positiveIndex)
#         print(index)
#         print(diff[positiveIndex[index]])
        diff[positiveIndex[index]] -= 1
        lagg += 1
        index += 1
        #print(diff, lagg)
    diff[np.where(diff < 0)] = 0
    dis = cap - diff
    #print(dis)
    return dis.astype(int).tolist()
# capacity = [10, 5, 0, 0]
# print(pattern_gen(10, capacity))

################################################
def pattern_weight(n, capacity):
    import numpy as np
    dist = [c*n/sum(capacity) for c in capacity]
    savings = 0
    newdist = []

    # each time taking random direction
    direction = np.random.binomial(size = 1, n = 1, p =0.5)[0]
    #print("direction: ", direction)

    # checking for fractions and setling up
    for value in dist if direction else dist[::-1]:
        newvalue = round(value)
        savings += value - newvalue
        if savings > 0.99:
            newvalue += 1
            savings -= 1
        elif savings < -0.99:
            newvalue -= 1
            savings += 1
        newdist.append(newvalue)

    #finalizing the direction
    newdist = newdist if direction else newdist[::-1]
    # print(savings)
    # print(newdist)
    return newdist        


# def defenseEval(attacked_Bus, selectedIDs, deceptiveIDs, verbose_ = True):

#     # reading attack matrix 
#     fName_ = "Attack_Space_"+str(NumOfBuses)+"_"+str(NumOfLines)+"_"+str(attacked_Bus)+".csv"
#     Attack_Data = np.genfromtxt("Attack Data//"+fName_, delimiter=',')
#     NumOfZ = NumOfBuses + NumOfLines * 2
#     numOfAttack = int (Attack_Data.shape[0] / (NumOfZ+1))


#     AttackEval = []
#     successCount = 0
#     totalAttackedSensors = 0
#     endingIndx = 0
    
#     for attackID in range(numOfAttack):

#         if verbose_: print("\nattackID: ", attackID)
#         startingIndx = endingIndx
#         endingIndx = (attackID+1) * (NumOfZ +1)

#         #reading each attack data
#         FData = Attack_Data[startingIndx:endingIndx,:].copy() 
#         FData[:,2] = FData[:,2] + Z_dec[:,2]
#         attackedIndx = np.where(FData[:,1]==1)[0]
#         if verbose_: print("Expected to attack: ", attackedIndx)
        
#         # Z_attack matrix to be sent to the EMS
#         Z_att = Z_dec.copy()
#         Z_att[attackedIndx, 2] = FData[attackedIndx, 2].copy()
        
#         #*************************************************************************
#         ###############################################################

#         #Z_rec = fixID(Z_att.copy(), selectedIDs, deceptiveIDs)

#         # defining the recovered matrix
#         Z_rec = Z_att.copy()
#         Z_rec[deceptiveIDs,0] = selectedIDs.copy()
#         Z_rec = Z_rec [Z_rec[:,0].argsort(kind = 'mergesort')]
#         #******************************************************

#         #mapping the attacked locations
#         actually_attacked = np.sort(mapOrgID(attackedIndx, selectedIDs, deceptiveIDs)).astype(int)

#         if verbose_: print("Actually attacked:", actually_attacked)

#         [States_check, Z_est_check, Z_mat_check, Rank_check, P_Noise_check] = SE_BDD(
#             np.copy(H_org), np.copy(Z_rec), Noise_th, Verbose = "False")

#         [M_Noise_check, P_Noise_check, Noisy_index_check] = Calc_Percent_of_Noise (
#             Z_est_check.copy(), Z_rec.copy(), Noise_th, Verbose = "False")
        
#         # detected noisy measurent
#         foundFDI_Idx =  sorted((set(Noisy_index_check)- set(Noisy_index_actu)) & set(actually_attacked))

#         # printing noisy indeces
#         if Noisy_index_check.size > 0 and verbose_ == True:
#             print("Noisy indeces before attack: ", Noisy_index_actu)
#             print("Noisy indeces after attack: ", Noisy_index_check)

#         # system Unobservable
#         if Rank_check < NumOfStates:
#             if verbose_: print("-----------  System Unobservable  ------------")
#             Deviation = -10
#             AttackEval.append((attackID+1, "unobservable", Deviation))
#         else:
#             # system is oversarble
#             # percent of deviation in the estiamted measurements
#             Deviation = np.linalg.norm(Z_est_check - Z_est_init)/np.linalg.norm(Z_est_init)*100
#             if Deviation > 100: Deviation = 100
                
#             ########################
#             totalAttackedSensors += np.sum(Z_rec[actually_attacked, 1])
#             ########################
#             successCount += len(set(Noisy_index_check) & set(actually_attacked))
#             #########################

#             # Detected as Bad Data
#             if len(foundFDI_Idx) > 0:       
#                 AttackEval.append((attackID+1, "detected", Deviation))
#                 #FData_list.append(FData)
#                 if verbose_: 
#                     print("!!!!!!!!!! Detected as Bad Data  !!!!!!!!!!!!")
#                     print("Detected Measurements: ", foundFDI_Idx)

#             # Successfull
#             else:
#                 if verbose_:print("$$$$$$$$$$$$$$  Successfull  $$$$$$$$$$$$$")
#                 AttackEval.append((attackID+1, "success", Deviation))


#     #np.savetxt("Eval_"+fName_+".csv", np.array(AttackEval), fmt= "%s",  delimiter=',')

#     # converting to matrix
#     successCount_avg = successCount / totalAttackedSensors * 100
#     AttackEval = np.array(AttackEval)
    
    
#     Attack_Summary = ((AttackEval[:,1]=='success').sum(),
#                   (AttackEval[:,1] == 'detected').sum(),  
#                   (AttackEval[:,1] == 'unobservable').sum())

#     histDatadeviation = np.histogram(AttackEval[:,2].astype(float), bins= np.arange(-20,101))
#     return AttackEval, Attack_Summary, histDatadeviation, successCount_avg

# ######################################################
# ###############################################33
def rand_ID (nodeX, n):
    import random 
    
    import numpy as np
    
    #print(f"Call at {nodeX.nodeType} {nodeX.nodeID }")
    
    if nodeX.leaf == True:
        #print("Reached Sensor:", nodeX.nodeID)
        nodeX.remSensor = 0
        return [nodeX.nodeID]
    
    remSensors = [nodeC.remSensor for nodeC in nodeX.child]
    
    #print ("remSensors: ", remSensors, "n = ", n)
    #creating pattern based on the remaining sensors
    pattern = pattern_weight(n, remSensors)
    #print("pattern: ", pattern)
    
    # call each child for the assigned number of randIDs
    decIDs = []
    for nodeC, nc in zip(nodeX.child, pattern):
        if nc > 0: 
            #print(f"Calling the child again: {nodeC.nodeType} {nodeC.nodeID }") 
            decIDs += rand_ID (nodeC, nc)
            nodeX.updateNode()
    ###### randomizing ###
    random.shuffle(decIDs)        
    #print(f"Returning from {nodeX.nodeType} {nodeX.nodeID } :{decIDs}")
    return decIDs
####################################################

def randomizeID(nodeX):
    import numpy as np
    import random 

    
    #print(f"Before: {nodeX.remSensor}")

    #n_child = nodeX.nchild
    orgID = []
    decID = []
    
    ids_mat = np.array(nodeX.ids) 
    orgID += ids_mat[np.where(np.array(nodeX.deceptive) == 0)].tolist()
    #print("orgID: ", orgID)
    decID += ids_mat[np.where(np.array(nodeX.deceptive) == 0)].tolist()
    #print("decID: ", decID)
    orgID += ids_mat[np.where(np.array(nodeX.deceptive) == 1)].tolist()
    #print("orgID: ", orgID)

    #spliting the work though each child
    for (index, nodeC) in enumerate(nodeX.child):
 
        #print("Calling Child ID: ", index)
        # there is no deceptive measurements
        if nodeC.decSensor == 0:
            continue
            
        #if there is any do the following
        else:
            #print("Recurive Call")
            decID += rand_ID(nodeX, nodeC.decSensor)
            #print(decID)
    #random.shuffle(decID)
    #print(f"After: {nodeX.remSensor}")
    nodeX.resetNode() 
    return orgID, decID



#hubIDs = randomizeID(hub[0])