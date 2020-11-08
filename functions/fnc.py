class Node:
    
    def __init__(self):
        self.nodeType = ''
        self.nodeID = None
        self.parent = None
        self.leaf = False
        self.totSensor = 0
        self.decSensor = 0
        self.remSensor = 0
        self.nchild = 0
        self.child = []
        self.ids = []
        self.reported = []
        self.deceptive = []
        self.values = []
        
    def addChild(self, child_node):
        if self.leaf == False:
            child_node.parent = self
            self.nchild += 1
            self.totSensor += child_node.totSensor
            self.decSensor += child_node.decSensor
            self.remSensor += child_node.remSensor           
            self.child.append(child_node)
            self.ids += child_node.ids
            self.reported += child_node.reported
            self.deceptive += child_node.deceptive
            self.values += child_node.values
            
        else:
            print("This is aleady a leaf node!")
            
    def updateNode(self):
        if self.leaf == False:
            self.remSensor = sum([ x.remSensor for x in self.child])
            #self.decSensor = sum([ x.decSensor for x in self.child])
        
    def resetNode(self):
        if self.leaf == True:
            self.remSensor = self.decSensor
            return
        else:
            for nodeC in self.child:
                nodeC.resetNode()
                nodeC.updateNode()
                
        self.updateNode()
    
        
    def printNode(self):
        print(
        "\nnodeType: ", self.nodeType,
        "\nnodeID: ", self.nodeID,
        "\nparent: ", self.parent,
        "\nleaf: ", self.leaf,
        "\ntotSensor: ", self.totSensor,
        "\ndecSensor: ", self.decSensor,
        "\nremSensor: ", self.remSensor,
        "\nnchild: ", self.nchild,
        "\nchild: ", self.child,
        "\nids: ", self.ids,
        "\nreported: ", self.reported,
        "\ndeceptive: ", self.deceptive,
        "\nvalues: ", self.values)
        
##################################################################################

def SE_BDD_(H_mat, Z_mat, W_list, Threshold_min, Threshold_max, Verbose = "True"):
    
    import numpy as np
    from numpy.random import seed
    from numpy.random import randn
    from numpy import mean
    from numpy import std

    if Verbose == "True": 
        print("\n **************  State Estimation ************* \n")

    doneFlag = 0
    fullRank = True
    Threshold = Threshold_min
    Threshold_step = 2
    numberOfMeasurements = H_mat.shape[0]
    numberOfStates = H_mat.shape[1]
    
    ############# starting State Estimation ###############
    #Z_mat = Z_mat[Z_mat[:, 0].argsort(kind='mergesort')]
    Z_mat_Copy = Z_mat.copy()
    W_list = np.array(W_list)

    # Space holder for Z_est
    #Z_est = np.zeros(numberOfMeasurements)
    #list_of_removed_sensors = []
    
    ################# Starting the loop ###################################
    while(doneFlag == 0):
        #list_of_removed_sensors = []
        #considering only the taken measurements
        
        consideredIndx = np.where(Z_mat[:,1] == 1)[0]
        #print(Z_mat)
        
        Z_msr = Z_mat[consideredIndx][:,2]

        # considering only the corresponding columns in H
        H_msr = H_mat[consideredIndx]
        
        # Measurement Covarriance Residual Matrix
        R_msr = np.diag(W_list[consideredIndx]) 
        #print(R_msr)
        R_inv = np.linalg.inv(R_msr)
        #print(R_inv)
        #Chekcing rank of H
        Rank = np.linalg.matrix_rank(H_msr) if H_msr.shape[0] > 0 else 0

        if Verbose == "True": 
            print("Current Rank", Rank)
            print("H_msr Shape: ", H_msr.shape)

        ###### H is full rank --> Start Estimating          
        if Rank == numberOfStates:
            
            #Estimating the states using WMSE estimator
            inv__Ht_Rinv_H__Ht = np.linalg.inv(H_msr.T@R_inv@H_msr)@H_msr.T
            States = inv__Ht_Rinv_H__Ht@R_inv@Z_msr
            
            # Ω = R − H. (H_T.R−1.H)−1.HT
            # Omega is a residual covarience matrix
            
            Omega_mat = R_msr - (H_msr@inv__Ht_Rinv_H__Ht)
            
            #print("Check :\n", R_msr - (H_msr@np.linalg.inv(H_msr.T@R_inv@H_msr)@H_msr.T))
            #print(f"R_msr: {R_msr} \n Shape {R_msr.shape}")
            #print(f"Omega_mat: {Omega_mat} \n Shape {Omega_mat.shape}")
                
            # Estimating the measurement from the estimated states
            
            Z_est = H_mat@States
            
            
            if Verbose == "True":
                print("\n Initital Z_m: "+str(Z_mat))
                print("\n Sates: \n"+str(States))
                print("\n Z_est: \n"+str(Z_est))
                print("Calling Noise.. CheckNoise...")

            #####################  Checking for Bad Data ##################################
            # Calculating the Noise
            M_Noise, P_Noise, doneFlag = CheckNoise(
                Z_est, Z_mat, Omega_mat, R_msr, Threshold, fullRank,  Verbose =="False")
        
        # H is not a full rank matrix ............. abort estimation 
        else:
            if Threshold < Threshold_max:
                Threshold += Threshold_step
                Z_mat = Z_mat_Copy.copy()
                if Verbose == "True": print(f"Relaxing the threshold to {Threshold}")
            else:
                doneFlag = -1 #system unobservable 
                if Verbose == "True": print(f"\n\n\nSystem Unobservable !, Rank = {Rank}")
                #####  Returning ##############
                fullRank = False
            
                Z_est = np.zeros(numberOfMeasurements)
                States = np.zeros(numberOfStates)
                M_Noise = np.zeros(numberOfMeasurements)
    ##############################################################################
    Noisy_Indx = np.where(Z_mat[:,1] == -1)[0]

    return States, Z_est, Z_mat, M_Noise, Noisy_Indx, fullRank, Threshold

##########################################################################################


def CheckNoise (Z_est, Z_mat, Omega_mat, R_msr, Threshold, fullRank,  Verbose = "False"):
    
    import math
    import numpy as np
    
    if fullRank != True:
    #         if Verbose == "True": 
    #             print("System Unobservable!"
        return None, None, Z_mat[:,0]        


    Z_msr = Z_mat[Z_mat[:, 1] == 1][:,2].copy()
    ####################################################   Here --------------------> 
    '''boolean index did not match indexed array along dimension 0; dimension is 53 
    but corresponding boolean dimension is 55'''
    if Verbose == "True":
        print("Z_est: ", Z_est.shape)
        print("Z_mat[:, 1] == 1", Z_mat[:, 1] == 1)
        print(Z_mat)

    Z_est_msr = Z_est[Z_mat[:, 1] == 1]
    
    # Calculating the measurement error
    
    M_Noise = (Z_msr - Z_est_msr)
    M_Noise_norm = M_Noise.copy()
    
    for index, _ in enumerate(M_Noise):
        if index == 0: continue
        try:
            M_Noise_norm [index] = np.absolute(M_Noise [index])/math.sqrt(Omega_mat[index, index])
        except:
            M_Noise_norm [index] = 0
            if Verbose == "True":
                print("index: ", index, np.absolute(M_Noise [index]))
                print(f" Value Error, Expected postive, Got {Omega_mat[index, index]}")
    
    Noise_mat_actual = np.zeros(Z_mat.shape[0])
    Noise_mat_actual[Z_mat[:,1] == 1] = M_Noise
    
    Noise_mat_norm = np.zeros(Z_mat.shape[0])
    Noise_mat_norm[Z_mat[:,1] == 1] = M_Noise_norm
    
    # Checking for Noisy data
    if np.max(Noise_mat_norm) > Threshold:
        targetedIndx = np.argmax(Noise_mat_norm)
        if Verbose == "True":
            print(f"targetedIndx in cut: {targetedIndx}--> Value : {Noise_mat_norm[targetedIndx]}")
            print("updating Z_mat...")
        Z_mat[targetedIndx, 1] = -1
        doneFlag = 0
    else:
        if Verbose == "True": print("No Bad Data Detected....")
        doneFlag = 1
        
    return Noise_mat_actual, Noise_mat_norm, doneFlag
##################################################################


##################################################################################
def SE_BDD_COR(H_mat, Z_mat, W_list, Threshold_min, Threshold_max, Corr, Verbose):
#     print("***********************************************")
    import numpy as np
    from numpy.random import seed
    from numpy.random import randn
    from numpy import mean
    from numpy import std

    if Verbose == "True": 
        print("\n **************  State Estimation ************* \n")

    doneFlag = 0
    fullRank = True
    Threshold = Threshold_min
    Threshold_step = 2
    numberOfMeasurements = H_mat.shape[0]
    numberOfStates = H_mat.shape[1]
    
    ############# starting State Estimation ###############
    #Z_mat = Z_mat[Z_mat[:, 0].argsort(kind='mergesort')]
    Z_mat_Copy = Z_mat.copy()
    W_list = np.array(W_list)

    # Space holder for Z_est
    #Z_est = np.zeros(numberOfMeasurements)
    #list_of_removed_sensors = []
    
    ################# Starting the loop ###################################
    while(doneFlag == 0):
        #list_of_removed_sensors = []
        #considering only the taken measurements
        
        consideredIndx = np.where(Z_mat[:,1] == 1)[0]
        #print(Z_mat)
        
        Z_msr = Z_mat[consideredIndx][:,2]

        # considering only the corresponding columns in H
        H_msr = H_mat[consideredIndx]
        
        # Measurement Covarriance Residual Matrix
        R_msr = np.diag(W_list[consideredIndx]) 
        #print(R_msr)
        R_inv = np.linalg.inv(R_msr)
        #print(R_inv)
        #Chekcing rank of H
        Rank = np.linalg.matrix_rank(H_msr) if H_msr.shape[0] > 0 else 0

        if Verbose == "True": 
            print("Current Rank", Rank)
            print("H_msr Shape: ", H_msr.shape)

        ###### H is full rank --> Start Estimating          
        if Rank == numberOfStates:
            
            #Estimating the states using WMSE estimator
            inv__Ht_Rinv_H__Ht = np.linalg.inv(H_msr.T@R_inv@H_msr)@H_msr.T
            States = inv__Ht_Rinv_H__Ht@R_inv@Z_msr
            
            # Ω = R − H. (H_T.R−1.H)−1.HT
            # Omega is a residual covarience matrix
            
            Omega_mat = R_msr - (H_msr@inv__Ht_Rinv_H__Ht)
            
            #print("Check :\n", R_msr - (H_msr@np.linalg.inv(H_msr.T@R_inv@H_msr)@H_msr.T))
            #print(f"R_msr: {R_msr} \n Shape {R_msr.shape}")
            #print(f"Omega_mat: {Omega_mat} \n Shape {Omega_mat.shape}")
                
            # Estimating the measurement from the estimated states
            
            Z_est = H_mat@States
            
            
            if Verbose == "True":
                print("\n Initital Z_m: \n"+str(Z_mat))
                print("\n Sates: \n"+str(States))
                print("\n Z_est: \n"+str(Z_est))
                print("Calling Noise.. CheckNoise...")

            #####################  Checking for Bad Data ##################################
            # Calculating the Noise
            M_Noise, P_Noise, doneFlag = CheckNoiseCor(
                Z_est, Z_mat, Omega_mat, R_msr, Threshold,  fullRank, Corr,  Verbose)
        
        # H is not a full rank matrix ............. abort estimation 
        else:
            if Threshold < Threshold_max:
                Threshold += Threshold_step
                Z_mat = Z_mat_Copy.copy()
                if Verbose == "True": print(f"Relaxing the threshold to {Threshold}")
            else:
                doneFlag = -1 #system unobservable 
                if Verbose == "True": print(f"\n\n\nSystem Unobservable !, Rank = {Rank}")
                #####  Returning ##############
                fullRank = False
            
                Z_est = np.zeros(numberOfMeasurements)
                States = np.zeros(numberOfStates)
                M_Noise = np.zeros(numberOfMeasurements)
    ##############################################################################
    Noisy_Indx = np.where(Z_mat[:,1] == -1)[0]

    return States, Z_est, Z_mat, M_Noise, Noisy_Indx, fullRank, Threshold

##########################################################################################


def CheckNoiseCor (Z_est, Z_mat, Omega_mat, R_msr, Threshold,  fullRank, Corr, Verbose):

    import math
    import numpy as np
    
    if fullRank != True:
    #         if Verbose == "True": 
    #             print("System Unobservable!"
        return None, None, Z_mat[:,0]        


    Z_msr = Z_mat[Z_mat[:, 1] == 1][:,2].copy()
    ####################################################   Here --------------------> 
    '''boolean index did not match indexed array along dimension 0; dimension is 53 
    but corresponding boolean dimension is 55'''
    
    if Verbose == "True":
        print("Starting BDD")
        print("Z_est: ", Z_est.shape)
        #print("Z_mat[:, 1] == 1", Z_mat[:, 1] == 1)
        #print(Z_mat)

    Z_est_msr = Z_est[Z_mat[:, 1] == 1]
    
    # Calculating the measurement error
    
    M_Noise = (Z_msr - Z_est_msr)
    M_Noise_norm = M_Noise.copy()
    
    # Calculating the normalized residuals
    for index, _ in enumerate(M_Noise):
        if index == 0: continue
        try:
            M_Noise_norm [index] = np.absolute(M_Noise [index])/math.sqrt(Omega_mat[index, index])
        except:
            M_Noise_norm [index] = 0
            if Verbose == "True":
                print("index: ", index, np.absolute(M_Noise [index]))
                print(f" Value Error, Expected postive, Got {Omega_mat[index, index]}")
    
#     Noise_mat_actual = np.zeros(Z_mat.shape[0])
#     Noise_mat_actual[Z_mat[:,1] == 1] = M_Noise
#     Noise_mat_norm = np.zeros(Z_mat.shape[0])
#     Noise_mat_norm[Z_mat[:,1] == 1] = M_Noise_norm

#     print(M_Noise.shape)

    Noise_mat_actual = M_Noise.copy()
    Noise_mat_norm = M_Noise_norm.copy()
    
    active_idx = np.where(Z_mat[:,1] == 1)[0]
    
    # Checking for Noisy data
    if np.max(Noise_mat_norm) > Threshold:
        tIndx = np.argmax(Noise_mat_norm)
        if Verbose == "True":
            print(f"targetedIndx in cut: {tIndx}--> Value : {Noise_mat_norm[tIndx]}")
            print("Updating Z_mat...")
            print("Before: ", Z_mat[tIndx])
        #print("R_msr: ", R_msr.shape, Omega_mat.shape)
        if Corr == True:
            Z_mat[active_idx[tIndx], 2] = Z_mat[active_idx[tIndx], 2] - R_msr[tIndx,tIndx]/Omega_mat[tIndx,tIndx]*M_Noise[tIndx]
        else:
            Z_mat[active_idx[tIndx], 1] = -1

        doneFlag = 0
        
        if Verbose == "True":
            print("After: ", Z_mat[tIndx])
    else:
        if Verbose == "True": print("No Bad Data Detected....")
        doneFlag = 1
    
        Noise_mat_actual = np.zeros(Z_mat.shape[0])
        Noise_mat_actual[Z_mat[:,1] == 1] = M_Noise.copy()
        Noise_mat_norm = np.zeros(Z_mat.shape[0])
        Noise_mat_norm[Z_mat[:,1] == 1] = M_Noise_norm.copy()
    ##############################################
    
    return Noise_mat_actual, Noise_mat_norm, doneFlag
##################################################################

##################################################################


def SE_BDD(H_mat, Z_mat, Threshold, Verbose = "True"):
    import numpy as np
    from numpy.random import seed
    from numpy.random import randn
    from numpy import mean
    from numpy import std

    #print("\n **************  State Estimation ************* \n")
    # takes H matrix Z measurements, Threshold and Verbose as input
    std_cutOff_th = 1.5
    Max_Threshold = 10
    Threshold_step = 2

    doneFlag = 0
    fullRank = True
    numberOfMeasurements = H_mat.shape[0]
    numberOfStates = H_mat.shape[1]
    #print("Considering only the taken measurements")

    ############# starting State Estimation ###############
    Z_mat = Z_mat[Z_mat[:,0].argsort(kind='mergesort')]
    Z_mat_Copy = Z_mat.copy()

    # Space holder for Z_est
    Z_est = np.zeros(numberOfMeasurements)
    list_of_removed_sensors = []
    while(doneFlag == 0):
        #list_of_removed_sensors = []
        #considering only the taken measurements
        Z_msr = Z_mat[Z_mat[:,1]==1][:,2]

        # considering only the corresponding columns in H
        H_msr = H_mat[Z_mat[:,1]==1]

        #Chekcing rank of H
        Rank = np.linalg.matrix_rank(H_msr)

        if Verbose == "True": 
            print("Current Rank", Rank)
            print("H_msr Shape: ", H_msr.shape)


        ###### H is full rank --> Start Estimating          
        if Rank == numberOfStates:

            #Estimating the states using WMSE estimator
            States = (np.linalg.inv(H_msr.T@H_msr)@H_msr.T)@Z_msr

            # Estimating the measurement from the estimated states
            Z_est = H_mat@States

            if Verbose == "True":
                pass
                #print("\n Initital Z_m: "+str(Z_mat))
                #print("\n Sates: \n"+str(States))
                #print("\n Z_est: \n"+str(Z_est))

            #####################  Checking for Bad Data ##################################
            # Calculating the Noise
            M_Noise, P_Noise, Noisy_Index = Calc_Percent_of_Noise (Z_est, Z_mat, Threshold, fullRank,  Verbose =="False")

            #####   If there are some noisy data  ############# 
            if( len(Noisy_Index) > 0): 

                #print("***************** Bad Data Detected *********************")
                #changing the noisy measurement parameter (taken) from 1 to 0

                # Analyzig the percentage of noise
                Noise_mean, Noise_std = mean(P_Noise), std(P_Noise)

                # identify outliers
                Noise_cut_off = Noise_mean + Noise_std * std_cutOff_th

                indxtbremoved = np.where(P_Noise > Noise_cut_off)[0]

                target_noisy_indx = indxtbremoved if indxtbremoved.size > 0 else  Noisy_Index #np.argsort(P_Noise)[::-1]


                if Verbose == "True": 
                    print(f"Noise_mean: {Noise_mean}, Noise_std: {Noise_std}")
                    print("Removing sensor: ", target_noisy_indx)

                #############################################################
                # removing noisy sensors from the state estimation process
                Z_mat[target_noisy_indx,1] = -1
                list_of_removed_sensors = list_of_removed_sensors + target_noisy_indx.tolist()

            else:
                if Verbose == "True": print("!!!! Done !!!\nNo more noisy data")
                doneFlag = 1
        else:
            if Threshold < Max_Threshold:

                Threshold += Threshold_step
                if Verbose == "True": print(f"Relaxing the nosie threshold to {Threshold}")
                list_of_removed_sensors = []
                Z_mat = Z_mat_Copy.copy()


            else:

                doneFlag = -1
                if Verbose == "True": print(f"\n\n\nSystem Unobservable !, Rank = {Rank}")

                #####  Returning ##############
                fullRank = False
                Z_est = np.zeros(numberOfMeasurements)
                States = np.zeros(numberOfStates)

    ############### Returning ########################
    #print("list_of_removed_sensors:", list_of_removed_sensors)
    return States, Z_est, Z_mat, fullRank
#############################################################################

def Calc_Percent_of_Noise (Z_est, Z_mat, Threshold, fullRank, Verbose):
    import numpy as np

    if fullRank != True:
#         if Verbose == "True": 
#             print("System Unobservable!"
        #print("System Unobservable!\n Returning Null")
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

#################################################################################
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
    #print("Rank of Noisy: " , Rank)
    
    Z_est = np.zeros(H_mat.shape[0])
    
    if Rank == H_mat.shape[1]:
        
        fullRank = True
        # Estimating the states using WMSE estimator
        #States_init = np.linalg.pinv(H_mat_init)@Z_msr_init
        States_init = (np.linalg.inv(H_mat_init.T@H_mat_init)@H_mat_init.T)@Z_msr_init

        # Estimating the measurement from the estimated states
        Z_est = H_mat@States_init
        ################################
#         print("SE_BDD_Noise - Z_est:\n", Z_est)
#         print("Mag: ", np.linalg.norm(Z_est))
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
        
    return States_init, Z_est, Z_mat, fullRank
###############################################
# def SE_BDD(H_mat, Z_mat, Threshold, Verbose):
#     import numpy as np
#     Z_mat = Z_mat[Z_mat[:,0].argsort(kind='mergesort')]
#     Z_mat_Copy = Z_mat.copy()
#     finalCheck = 1
#     #print("\n **************  State Estimation ************* \n")
#     #takes H matrix Z measurements, Threshold and number of Equations as input
#     #print("Considering only the taken measurements")
#     Z_msr_init = Z_mat[Z_mat[:,1]==1][:,2]
    
#     # considering only the corresponding columns in H   
#     H_mat_init = H_mat[Z_mat[:,1]==1]

#     #printing the sizes of H, Rank (H) and Z
#     Rank=np.linalg.matrix_rank(H_mat_init)
    
#     if Rank == H_mat.shape[1]:
        
#         # Estimating the states using WMSE estimator
#         #States_init = np.linalg.pinv(H_mat_init)@Z_msr_init
#         States_init = (np.linalg.inv(H_mat_init.T@H_mat_init)@H_mat_init.T)@Z_msr_init

#         # Estimating the measurement from the estimated states
#         Z_est = H_mat@States_init

#         if Verbose == "True":
#             print("\n Initial H: "+str(H_mat_init.shape))
#             print("\n Rank H: "+str(Rank))
#             print("\n Initital Z_m: "+str(Z_msr_init.shape))
#             print("\n Sates: \n"+str(States_init))
#             print("\n Z_est: \n"+str(Z_est))


#         # Calculating the Noise
#         [M_Noise, P_Noise, Noisy_index] = Calc_Percent_of_Noise (Z_est.copy(), Z_mat.copy(), Threshold, Verbose)

#         if(len(Noisy_index)>0): 

#             #print("***************** Bad Data Detected *********************")

#             #changing the noisy measurement parameter (taken) from 1 to 0
            
            
#             noisyIdxsorted = np.argsort(P_Noise)[::-1]
            
#             print("Removing sensor: ", noisyIdxsorted[0])
            
#             #############################################################
#             temp=Z_mat[:,1].copy()
#             temp[noisyIdxsorted[0]]= -1
#             Z_mat[:,1]=temp.copy()
            

#             # considering only the corresponding columns in H   
#             #printing the sizes of H, Rank (H) and Z
#             H_mat_check = H_mat[Z_mat[:,1]==1]
#             RankCheck=np.linalg.matrix_rank(H_mat_check)
            
            
#             if Verbose == "True":
#                 print("\nModified Z: \n"+str(Z_mat))
#                 print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
#             if RankCheck == H_mat.shape[1]:
#                 [States_init, Z_est, Z_mat, Rank, P_Noise]=SE_BDD (H_mat,Z_mat, Threshold, Verbose)
            
#             else: 
#                 #print("Cant remove this noisy measurement! Singularity warning!!")
#                 Rank = RankCheck
#             #print("***************** End of SE **********************")
#     else:
#         #print("The H is not a full rank matrix !!")
#         Rank = Rank
    
#     if Verbose != "Final":
#         #print("Final Checcking...")
#         Verbose = "Final"
#         Z_mat = Z_mat_Copy.copy()
#         [M_Noise, P_Noise, Noisy_index] = Calc_Percent_of_Noise (Z_est.copy(), Z_mat.copy(), Threshold, Verbose)
#         #print("Noisy_index", Noisy_index)
#         #print("Z_mat: ", Z_mat)
#         if Noisy_index.size>0: 
#             Z_mat[Noisy_index, 1] = -1 
            
#         [States_init, Z_est, Z_mat, Rank, P_Noise] = SE_BDD (H_mat, Z_mat, Threshold, Verbose)
#         ######################################
# #         [M_Noise_dece, P_Noise_dece, Noisy_index_dece] = Calc_Percent_of_Noise (
# #             Z_est_dece.copy(), Z_dec.copy(), Noise_th, Verbose = True)

# #         Z_dec_new = Z_dec.copy()

# #         if Noisy_index_dece.size > 0:
# #             print("Noisy_index_dece :", Noisy_index_dece)
# #             Z_dec_new[Noisy_index_dece, 1] = -1 
        
#     return States_init, Z_est, Z_mat, Rank, P_Noise

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
    import numpy as np
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
    import re
    
    while(True):
        # reading the line
        if lineID >= len(netdata):
            print("Missing data")
            return
        #print(lineID,netdata[lineID])
        line = netdata[lineID]
        #print(line, netdata[lineID])
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
                    #xList.append([int(i) for i in line.strip().split(", ")]) 
                    xList.append([int(i) for i in re.split(r'[;,\s]\s*', line.strip())]) 
                    
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


######################################################
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
    #random.shuffle(decIDs)
    #print(f"Returning from {nodeX.nodeType} {nodeX.nodeID } :{decIDs}")
    return decIDs
####################################################


def randomizeID(nodeX, randType):
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

    if randType == 'random':
        temp_orgIDs = ids_mat[np.where(np.array(nodeX.deceptive) == 1)].tolist()
        random.shuffle(temp_orgIDs)
        decID += temp_orgIDs
    
    #spliting the work though each child
    elif randType == 'tree':
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
        
    # returning results
    return orgID, decID

###########################


####################################################
# def randomizeID(nodeX):
#     import numpy as np
#     import random 

    
#     #print(f"Before: {nodeX.remSensor}")

#     #n_child = nodeX.nchild
    
#     orgID = []
#     decID = []
    
#     ids_mat = np.array(nodeX.ids) 
#     orgID += ids_mat[np.where(np.array(nodeX.deceptive) == 0)].tolist()
#     #print("orgID: ", orgID)
#     decID += ids_mat[np.where(np.array(nodeX.deceptive) == 0)].tolist()
#     #print("decID: ", decID)
#     orgID += ids_mat[np.where(np.array(nodeX.deceptive) == 1)].tolist()
#     #print("orgID: ", orgID)

#     #spliting the work though each child
#     for (index, nodeC) in enumerate(nodeX.child):
 
#         #print("Calling Child ID: ", index)
#         # there is no deceptive measurements
#         if nodeC.decSensor == 0:
#             continue
            
#         #if there is any do the following
#         else:
#             #print("Recurive Call")
#             decID += rand_ID(nodeX, nodeC.decSensor)
#             #print(decID)
#     #random.shuffle(decID)
#     #print(f"After: {nodeX.remSensor}")
#     nodeX.resetNode() 
#     return orgID, decID

# ###########################
def revsoftMax(data_):
    data = [-x/max(data_) for x in data_]
    return softMax(data)


# ###########################
def revsoftMaxx(data_):
#     data = [-x/max(data_) for x in data_]
    return softMax(1-softMax(data_))

def softMax(data):
    import math
    import numpy as np
#     print("input:", data)
    try:
        data = data/data.max()
    except:
        pass

    exp_data = [math.exp(x) for x in data]
    probability = [x/sum(exp_data) for x in exp_data]
#     print("output:", probability)
    return np.array(probability)
##################################

def func_dec_view(IDbank):
    ID_bank = []
    org_view = IDbank[0][0].copy()
    dec_view = IDbank[0][1].copy()
    ID_bank.append((org_view, dec_view))
    
    for iview in range(len(IDbank)-1): 
        #print('\n\nLevel: ',iview+1)
        dec_new = []
        
        for i in  range(len(dec_view)):
            for j in range(len(dec_view)):
                
                if  dec_view[i] == IDbank[iview+1][0][j]:
                    dec_new.append(IDbank[iview+1][1][j])
        dec_view = dec_new.copy()
        ID_bank.append((org_view, dec_view))
    return ID_bank