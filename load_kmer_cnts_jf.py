#!/usr/bin/env python3
#~/miniconda3/bin/python3

import gzip
import pandas as pd
import numpy as np
from numpy import array
import ntpath

from Bio import SeqIO
from glob import glob
from itertools import product

from functools import partial
from multiprocessing import Pool
import os.path
import pickle

from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RepeatedStratifiedKFold

def load_kmers(kmer_size,data_sets, allowed_labels=['0','1']):
    
    # Feng, Backhed, Ferretti, Hadza, HMP1-2, Karlsson_2013, LeChatelier, MetaHIT, Nielsen, Peru, RA, Raymond, Yassour, Fiji, HMP, IGC, Karlsson_2013_no_adapter, LiverCirrhosis, Mongolian, Qin_et_al, RA_no_adapter, Twins, Zeller_2014

    kmer_cnts=[]
    accessions=[]
    labels=[]
    metadata=load_metadata()
    domain_labels=[] # this keeps track of which domain we have

    domain_label=1
    for data_set in data_sets:
        print(data_set)
        input_dir = os.path.expanduser('~/deep_learning_microbiome/data/%smers_jf/%s') %(kmer_size, data_set)
        file_pattern='*.gz'
        files=glob(input_dir + '/' + file_pattern)

        for inFN in files:        
            run_accession=inFN.split('/')[-1].split('_')[0]
            if run_accession in metadata.keys():
                label=metadata[run_accession]
                if label in allowed_labels:
                    
                    file = gzip.open(inFN, 'rb')
                
                    new_cnts=[]
                    for line in file:   
                        new_cnts.append(float(line.decode('utf8').strip('\n')[1:]))
                    new_cnts=np.asarray(new_cnts)
                    
                    # some entries are zero for various bioinformatic reasons. Exclude these so that they don't mess up training/testing
                    if new_cnts.sum()>0:
                        kmer_cnts.append(new_cnts)
                        labels.append(metadata[run_accession])
                        accessions.append(run_accession)
                        domain_labels.append(domain_label)
                        
        
        domain_label +=1

    kmer_cnts=np.asarray(kmer_cnts)
    
    # one hot encode the domain_labels
    domain_labels = onehot_encode(domain_labels)

    return kmer_cnts, accessions, labels, domain_labels
    

def onehot_encode(labels):
    
    values = array(labels)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return onehot_encoded

    

def load_metadata():
    metadata={} # run_accession -> disease_status
    directory=os.path.expanduser('~/deep_learning_microbiome/data/metadata')
    
    exclude = []
    with open(os.path.expanduser("~/deep_learning_microbiome/scripts/exclude.txt")) as text:
        for line in text:
            line = line.rstrip("\n")
            line = line.strip("'")
            exclude.append(line)

    include_pasolli = []
    with open(os.path.expanduser("~/deep_learning_microbiome/scripts/include_pasolli.txt")) as text:
        for line in text:
            line = line.rstrip("\n")
            line = line.strip("'")
            include_pasolli.append(line)
                

    # Qin et al. T2D data:

    qin_et_al_inFN='%s/Qin_2012_ids_all.txt' %directory
    qin_et_al_file=open(qin_et_al_inFN, 'r')
    for line in qin_et_al_file:
        items=line.strip('\n').split('\t')
        name=items[0]
        run_accession=items[2]
        disease_status=items[5]
        if name in include_pasolli:
            metadata[run_accession] = disease_status

    # Zhang et al. RA data:

    RA_inFN='%s/arthritis_metaData_merged.txt' %directory
    RA_file=open(RA_inFN, 'r')
    for line in RA_file:
        items=line.strip('\n').split('\t')
        run_accession=items[2]
        disease_status=items[3]
        if run_accession not in exclude and run_accession != 'run_accession':
            metadata[run_accession] = disease_status

    # MetaHIT IBD data:
    
    MetaHIT_inFN='%s/MetaHIT_ids.txt' %directory
    MetaHIT_file=open(MetaHIT_inFN, 'r')
    for line in MetaHIT_file:
        items=line.strip('\n').split('\t')
        run_accession=items[1]
        disease_status=items[5]
        if run_accession not in exclude:
            metadata[run_accession] = disease_status
    

    # HMP1-2 data (everyone is healthy):
    
    HMP_inFN='%s/HMP1-2_samples_only.txt' %directory
    HMP_file=open(HMP_inFN, 'r')
    for line in HMP_file:
        items=line.strip('\n').split('\t')
        run_accession=items[0]
        disease_status='0'
        if run_accession not in exclude:
            metadata[run_accession] = disease_status
    

    # Karlsson et al. (T2D European women)
    
    # there are two files. One matches the Run accession with the id. the other matches the id with the disease status. 
    Karlsson_inFN='%s/Karlsson.txt' %directory
    Karlsson_file=open(Karlsson_inFN, 'r')

    run_acc_inFN='%s/PRJEB1786.txt' %directory
    run_acc_file=open(run_acc_inFN, 'r')
    
    # read in run_acc into a dictionary:
    #key=id, value=run_acc
    run_acc_dict={}
    for line in run_acc_file:
        items=line.strip('\n').split('\t')
        run_accession=items[4]
        sample_id = items[6]
        run_acc_dict[sample_id]=run_accession
    
    for line in Karlsson_file:
        items=line.strip('\n').split('\t')
        sample_id=items[0]
        if sample_id != 'Sample ID':
            disease_status=items[2]
            run_accession=run_acc_dict[sample_id]
            #
            if disease_status == 'NGT':
                disease_status ='0'
            elif disease_status == 'T2D':
                disease_status ='1' # note that I'm collapsing T2D and Impaired Glucose Toleraance (IGT) into one group
                disease = 'T2D'
            else:
                exclude.append(run_accession)
            if run_accession not in exclude and sample_id in include_pasolli:
                metadata[run_accession] = disease_status
    


    # LiverCirrhosis 

    LiverCirrhosis_inFN='%s/LiverCirrhosis.txt' %directory
    LiverCirrhosis_file=open(LiverCirrhosis_inFN, 'r')
    for line in LiverCirrhosis_file:
        items=line.strip('\n').split('\t')
        sample_id=items[1]
        disease_status=items[6]
        if disease_status == 'N':
            disease_status ='0'
        else:
            disease_status ='1' 
        if sample_id not in exclude and sample_id in include_pasolli:
            metadata[sample_id] = disease_status
    

    #Feng_CRC
    Feng_CRC_inFN='%s/Feng_CRC.txt' %directory
    Feng_CRC_file=open(Feng_CRC_inFN, 'r')
    for line in Feng_CRC_file:
        items=line.strip('\n').split('\t')
        sample_id=items[7]
        disease_status=items[8]
        if disease_status == 'Stool sample from controls':
            disease_status ='0'
        else:
            disease_status ='1' 
        if sample_id not in exclude:
            metadata[sample_id] = disease_status

    #Zeller_CRC, France
    # 61 healthy, 27 small, 15 large adenocarcinmoas, 15,7,10,21 various stages of CRC
    # I am ignoring adenocarcinomas, just like Zeller et al. did. 
    
    Zeller_inFN='%s/Zeller_metadata.txt' %directory
    Zeller_file=open(Zeller_inFN, 'r')
    Zeller_file.readline() #header
    for line in Zeller_file:
        items=line.strip('\n').split('\t')
        sample_id=items[1]
        disease_status=items[12]
        if disease_status == 'Control':
            disease_status ='0'
        elif disease_status == 'CRC':
            disease_status ='1' 
        if sample_id not in exclude and disease_status != 'NA' and sample_id in include_pasolli:
            metadata[sample_id] = disease_status

    #LeChatelier
    LeChatelier_inFN='%s/LeChatelier_metadata.txt' %directory 
    LeChatelier_file=open(LeChatelier_inFN, 'r')  
    for line in LeChatelier_file:
        items=line.strip('\n').split('\t')  
        sample_id=items[0]
        BMI=float(items[1])
        if BMI <=25:
            disease_status='0'
        elif BMI > 30:
            disease_status='1'
        else:
            disease_status='NA'

        if sample_id not in exclude and disease_status != 'NA': 
            metadata[sample_id] = disease_status

    #Backhed
    Backhed_inFN='%s/Backhed_mothers_only_accessions.txt' %directory
    Backhed_file=open(Backhed_inFN, 'r')
    for line in Backhed_file:
        items=line.strip('\n').split('\t')
        sample_id=items[0]
        disease_status='0' 
        if sample_id not in exclude and disease_status != 'NA':
            metadata[sample_id] = disease_status  

    #Ferretti
    Ferretti_inFN='%s/PRJNA352475_Mothers_Fecal_accessions.txt' %directory
    Ferretti_file=open(Ferretti_inFN, 'r')
    for line in Ferretti_file:
        items=line.strip('\n').split('\t')
        sample_id=items[0]
        disease_status='0' 
        if sample_id not in exclude and disease_status != 'NA':
            metadata[sample_id] = disease_status  

    #Fiji
    Fiji_inFN='%s/PRJNA217052_stool_only.txt' %directory
    Fiji_file=open(Fiji_inFN, 'r') 
    for line in Fiji_file: 
        items=line.strip('\n').split('\t')
        sample_id=items[6]
        disease_status='0'
        if sample_id not in exclude and disease_status != 'NA':
            metadata[sample_id] = disease_status

    #Hadza
    Hadza_inFN='%s/Hadza_sample_ids.txt' %directory
    Hadza_file=open(Hadza_inFN, 'r') 
    for line in Hadza_file: 
        items=line.strip('\n').split('\t')
        sample_id=items[0]
        disease_status='0'
        if sample_id not in exclude and disease_status != 'NA':
            metadata[sample_id] = disease_status

    #IGC
    IGC_inFN='%s/PRJEB5224.txt' %directory
    IGC_file=open(IGC_inFN, 'r')
    for line in IGC_file:
        items=line.strip('\n').split('\t')
        sample_id=items[5]
        disease_status='0' 
        if sample_id not in exclude and disease_status != 'NA':
            metadata[sample_id] = disease_status  

    #Mongolian
    Mongolian_inFN='%s/PRJNA328899_run_accessions_only.txt' %directory
    Mongolian_file=open(Mongolian_inFN, 'r')
    for line in Mongolian_file:
        items=line.strip('\n').split('\t')
        sample_id=items[0]
        disease_status='0' 
        if sample_id not in exclude and disease_status != 'NA':
            metadata[sample_id] = disease_status  
    
    #Nielsen
    Nielsen_inFN='%s/Nielsen_2014_sample_ids_relevant.txt' %directory
    Nielsen_file=open(Nielsen_inFN, 'r')
    for line in Nielsen_file:
        items=line.strip('\n').split('\t')
        sample_id=items[0]
        disease_status='0' 
        if sample_id not in exclude and disease_status != 'NA':
            metadata[sample_id] = disease_status  

    #Peru
    Peru_inFN='%s/Peru_paired_end_accessions_only.txt' %directory
    Peru_file=open(Peru_inFN, 'r')
    for line in Peru_file:
        items=line.strip('\n').split('\t')
        sample_id=items[0]
        disease_status='0' 
        if sample_id not in exclude and disease_status != 'NA':
            metadata[sample_id] = disease_status  

    #Raymond
    Raymond_inFN='%s/Raymond_day0_patient_ids.txt' %directory
    Raymond_file=open(Raymond_inFN, 'r')
    for line in Raymond_file:
        items=line.strip('\n').split('\t')
        sample_id=items[0]
        disease_status='0' 
        if sample_id not in exclude and disease_status != 'NA':
            metadata[sample_id] = disease_status  

    #Twins
    Twins_inFN='%s/PRJEB9576_run_accessions_only.txt' %directory
    Twins_file=open(Twins_inFN, 'r')
    for line in Twins_file:
        items=line.strip('\n').split('\t')
        sample_id=items[0]
        disease_status='0' 
        if sample_id not in exclude and disease_status != 'NA':
            metadata[sample_id] = disease_status  

    #Yassour
    Yassour_inFN='%s/Yassour_mother_birth_accessions_only.txt' %directory
    Yassour_file=open(Yassour_inFN, 'r')
    for line in Yassour_file:
        items=line.strip('\n').split('\t')
        sample_id=items[0]
        disease_status='0' 
        if sample_id not in exclude and disease_status != 'NA':
            metadata[sample_id] = disease_status  

    return metadata

    



def load_single_disease(data_set, kmer_size, n_splits, n_repeats, precomputed_kfolds):

    data_sets=[data_set]
    allowed_labels=['0','1']
    kmer_cnts, accessions, labels, domain = load_kmers(kmer_size,data_sets, allowed_labels)

    labels=np.asarray(labels)
    labels=labels.astype(np.int)

    data=pd.DataFrame(kmer_cnts)
    data_normalized = normalize(data, axis = 1, norm = 'l1')
    

    # compute the indexes for stratified k fold:
    # I may have precomputed this so that we can  use the same idxs for different model. This is what pasolli did and also gives us more power to distinguish model perf. 

    if precomputed_kfolds==False:
        rskf=repeated_stratified_k_fold(data_normalized, labels, n_splits, n_repeats)
        # save to a pickle so that the same idxs can be used for multiple runs
        # to save, I need to convert this to a different data structure
        train_indexs=[]
        test_indexs=[]
        for train_index, test_index in rskf:
            train_indexs.append(train_index)
            test_indexs.append(test_index)
        
        directory=os.path.expanduser('~/deep_learning_microbiome/data/precomputed_kfolds')
        pickle.dump([train_indexs,test_indexs], open( "%s/%s_single_disease.p"  %(directory,data_set), "wb" ) )
        

    else:
        [train_indexs,test_indexs] = pickle.load(open( "%s/%s_single_disease.p" %(directory,data_set), "rb" ))

    return data_normalized, labels, [train_indexs,test_indexs]



def repeated_stratified_k_fold(data, labels, n_splits, n_repeats):
    # this will cut up the data so that healthy and disease labels are represented proportionately in test/training across all folds. 
    # when training on just 1 data set, we will use the healthy vs disease label for the skf.
    # when training the autoencoder on all data sets, we will use the labels of the dataset for the skf.
    # now, when we train a supervised learning model on two or more datasets, then we have to have unique labels for each dataset/disease class
    # I will deal with this latter case later on. 

   
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    rskf = rskf.split(data, labels)    
    
    # rskf has the indexes for training and testing:
    # for train_index, test_index in rskf:

    return rskf

