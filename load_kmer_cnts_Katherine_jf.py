#!/usr/bin/env python3
#~/miniconda3/bin/python3
'''
    This file was originally written by Annamarie. I copied it and modified it to load the data in a format that is compatible with my deep learning code.
'''



import gzip
import numpy as np
from numpy import array
import ntpath
 
from Bio import SeqIO
from glob import glob
from itertools import product

from functools import partial
from multiprocessing import Pool
import os.path

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def load_kmers(kmer_size, data_sets):
    kmer_cnts=[]
    accessions=[]
    labels=[]
    metadata=load_metadata()
    domain_labels=[] # this keeps track of which domain we have
    
    domain_label=1
    for data_set in data_sets:
        if data_set == "MetaHIT":
            kmer_cnts_metahit, labels_metahit = load_metahit_kmers(kmer_size)
            kmer_cnts_metahit = np.asarray(kmer_cnts_metahit)
            for i in range(len(kmer_cnts_metahit)):
                kmer_cnts.append(kmer_cnts_metahit[i])
                labels.append(labels_metahit[i])
        else:
            input_dir = os.path.expanduser('~/deep_learning_microbiome/data/%smers_jf/%s') %(kmer_size, data_set)
            file_pattern='*.gz'
            files=glob(input_dir + '/' + file_pattern)
            run_accessions = []
            for inFN in files:
                run_accession=inFN.split('/')[-1].split('_')[0]
                if run_accession in metadata.keys():
                    label=metadata[run_accession]
                    run_accessions.append(run_accession)
                    labels.append(metadata[run_accession])
                    accessions.append(run_accession)
                    domain_labels.append(domain_label)

                    file = gzip.open(inFN, 'rb')
                
                    new_cnts=[]
                    for line in file:
                        new_cnts.append(float(line.decode('utf8').strip('\n')[1:]))
                    new_cnts=np.asarray(new_cnts)
                    kmer_cnts.append(new_cnts)

        
        domain_label +=1

    kmer_cnts=np.asarray(kmer_cnts)
    return kmer_cnts, labels

def load_metahit_kmers(kmer_size):
    directory=os.path.expanduser('~/deep_learning_microbiome/data/metadata')
    run_accession_to_sample = {}
    sample_to_kmers = {}
    sample_to_labels = {}
    kmer_cnts = []
    metadata=load_metadata()
    
    with open(os.path.expanduser('%s/MetaHIT_ids.txt' %directory)) as text:
        for line in text:
            line = line.rstrip("\n")
            line = line.strip("'")
            fields = line.split('\t')
            run_accession = fields[2]
            sample_name = fields[0]
            if sample_name not in sample_to_kmers and run_accession in metadata:
                sample_to_kmers[sample_name] = []
                sample_to_labels[sample_name] = metadata[run_accession]
            run_accession_to_sample[run_accession] = sample_name

    input_dir = os.path.expanduser('~/deep_learning_microbiome/data/%smers_jf/%s') %(kmer_size, "MetaHIT")
    file_pattern='*.gz'
    files=glob(input_dir + '/' + file_pattern)
    for inFN in files:
        run_accession=inFN.split('/')[-1].split('_')[0]
        if run_accession in metadata:
            sample_name = run_accession_to_sample[run_accession]
            sample_kmer_cnts = sample_to_kmers[sample_name]
            file = gzip.open(inFN, 'rb')
            run_accession_cnts = []
            for line in file:
                run_accession_cnts.append(float(line.decode('utf8').strip('\n')[1:]))
            sample_kmer_cnts.append(run_accession_cnts)
            

    labels = []
    for sample in sample_to_kmers:
        labels.append(sample_to_labels[sample])
        kmer_cnts.append(np.sum(sample_to_kmers[sample], axis=0))

    return kmer_cnts, labels
    
    

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
    include_pasolli = []
    with open(os.path.expanduser("~/deep_learning_microbiome/scripts/exclude.txt")) as text:
        for line in text:
            line = line.rstrip("\n")
            line = line.strip("'")
            exclude.append(line)
            
    with open(os.path.expanduser("~/deep_learning_microbiome/scripts/include_pasolli.txt")) as text:
        for line in text:
            line = line.rstrip("\n")
            line = line.strip("'")
            include_pasolli.append(line)
                

    # Qin et al. T2D data:

    qin_et_al_inFN='%s/Qin_2012_ids_all.txt' %directory
    qin_et_al_file=open(qin_et_al_inFN, 'r')
    count = 0
    names = []
    run_accessions = []
    for line in qin_et_al_file:
        items=line.strip('\n').split('\t')
        name=items[0]
        run_accession=items[2]
        disease_status=items[5]
        if disease_status == '1':
            disease = "T2D"
        else:
            disease = "Healthy"
        if name in include_pasolli:
            run_accessions.append(run_accession)
            metadata[run_accession] = [disease_status, 'T2D', 'Qin_et_al', disease]

         

    # Zhang et al. RA data:

    RA_inFN='%s/arthritis_metaData_merged.txt' %directory
    RA_file=open(RA_inFN, 'r')
    for line in RA_file:
        items=line.strip('\n').split('\t')
        run_accession=items[2]
        disease_status=items[3]
        if disease_status == '1':
            disease = "RA"
        else:
            disease = "Healthy"
        if run_accession not in exclude and run_accession != 'run_accession':
            metadata[run_accession] = [disease_status, 'RA', 'RA', disease]

    # MetaHIT IBD data:
    
    MetaHIT_inFN='%s/MetaHIT_ids.txt' %directory
    MetaHIT_file=open(MetaHIT_inFN, 'r')
    names = []
    for line in MetaHIT_file:
        items=line.strip('\n').split('\t')
        name = str(items[1])
        name = name.replace(".", "_")
        name = name.replace("-", "_")
        run_accession=items[2]
        disease_status=items[5]
        if disease_status == '1':
            disease = "IBD"
        else:
            disease = "Healthy"
        if name in include_pasolli:
            metadata[run_accession] = [disease_status, 'IBD', 'MetaHIT', disease]


    # HMP data (everyone is healthy):
    
    HMP_inFN='%s/HMP_samples_314.txt' %directory
    HMP_file=open(HMP_inFN, 'r')
    for line in HMP_file:
        items=line.strip('\n').split('\t')
        run_accession=items[0]
        disease_status = '0'
        disease = 'Healthy'
        if run_accession not in exclude:
            metadata[run_accession] = [disease_status, 'Healthy', 'HMP', disease]
    

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
            
            if disease_status == 'NGT':
                disease_status ='0'
                disease = 'Healthy'
            elif disease_status == 'T2D':
                disease_status ='1' # note that I'm collapsing T2D and Impaired Glucose Toleraance (IGT) into one group
                disease = 'T2D'
            if sample_id in include_pasolli:
                metadata[run_accession] = [disease_status,  'T2D', 'Karlsson_2013', disease]



    # LiverCirrhosis 

    LiverCirrhosis_inFN='%s/LiverCirrhosis.txt' %directory
    LiverCirrhosis_file=open(LiverCirrhosis_inFN, 'r')
    for line in LiverCirrhosis_file:
        count +=1
        items=line.strip('\n').split('\t')
        sample_id=items[1]
        disease_status=items[6]
        if disease_status == 'N':
            disease_status ='0'
            disease = 'Healthy'
        else:
            disease_status ='1'
            disease = 'LC'
        if sample_id in include_pasolli:
            metadata[sample_id] = [disease_status,  'LC', 'LiverCirrhosis', disease]

    

    #Feng_CRC
    Feng_CRC_inFN='%s/Feng_CRC.txt' %directory
    Feng_CRC_file=open(Feng_CRC_inFN, 'r')
    for line in Feng_CRC_file:
        items=line.strip('\n').split('\t')
        sample_id=items[7]
        disease_status=items[8]
        if disease_status == 'Stool sample from controls':
            disease_status ='0'
            disease = 'Healthy'
        else:
            disease_status ='1'
            disease = 'CRC'
        if sample_id not in exclude:
            metadata[sample_id] = [disease_status,  'CRC', 'Feng', disease]

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
            disease = 'Healthy'
        elif disease_status == 'CRC':
            disease_status ='1'
            disease = 'CRC'
        if sample_id in include_pasolli:
            metadata[sample_id] = [disease_status,  'CRC', 'Zeller', disease]
    return metadata

if __name__ == "__main__":
    for kmer_size in [5]:
        #kmers, labels = load_kmers(kmer_size, ['HMP', 'Feng', 'Zeller_2014', 'RA', 'MetaHIT','LiverCirrhosis', 'Karlsson_2013', 'Qin_et_al'])
        for dataset in ['HMP', 'Feng', 'Zeller_2014', 'RA', 'MetaHIT','LiverCirrhosis', 'Karlsson_2013', 'Qin_et_al']:
            kmers, labels = load_kmers(kmer_size, [ dataset, ])
            total = kmers.shape[0]
            diseased = [labels[i][0] for i in range(len(labels))].count('1')
            print(dataset + ": " + str(total) + " samples, " + str(total - diseased) + " control, " + str(diseased) + " diseased")

