#run from inside directory with csv files
import pandas as pd
import numpy as np
import os
from glob import glob
from itertools import product
import pickle

# List of unique HMP gut sample ids from Nandita
# Some have been commented out due to unusual output (i.e. values < 10, and NaN)
hmp_ids_314 = [
    "700113954",
    "700023788",
    "700114653",
    "700172461",
    "700024509",
    "700015181",
    "700111439",
    "700163030",
    "700171066",
    "700024930",
    "700038263",
    "700024233",
    "700116778",
    "700113975",
    "700163811",
    "700173119",
    "700113762",
    "700034024",
    "700117172",
    "700164641",
    "700173377",
    "700102659",
    "700164050",
    "700023578",
    "700103446",
    "700033363",
    "700015981",
    "700100471",
    "700023919",
    "700106615",
    "700116358",
    "700106170",
    "700163772",
    "700102299",
    "700110419",
    "700122108",
    "700023845",
    "700103710",
    "700109987",
    "700032068",
    "700116505",
    "700024673",
    "700105771",
    "700116468",
    "700102043",
    "700171114",
    "700023634",
    "700114082",
    "700021306",
    "700037632",
    "700035861",
    "700108896",
    "700106876",
    "700171648",
    "700014562",
    "700110222",
    "700117000",
    "700165052",
    "700099803",
    "700119165",
    "700095524",
    "700116668",
    "700096865",
    "700035785",
    "700112490",
    "700111505",
    "700117031",
    "700164417",
    "700173086",
    "700023872",
    "700106809",
    "700037852",
    "700163699",
    "700171747",
    "700021824",
    "700038158",
    "700164450",
    "700016456",
    "700100600",
    "700119739",
    "700038072",
    "700034622",
    "700111156",
    "700038386",
    "700014954",
    "700109921",
    "700099886",
    "700172726",
    "700034926",
    "700112755",
    "700037042",
    "700103621",
    "700117828",
    "700171115",
    "700173483",
    "700116730",
    "700164994",
    "700171341",
    "700035533",
    "700102432",
    "700107189",
    "700111026",
    "700120753",
    "700024086",
    "700037539",
    "700171441",
    "700015415",
    "700101243",
    "700119562",
    "700021876",
    "700038472",
    "700113616",
    "700037738",
    "700164611",
    "700109449",
    "700171390",
    "700034081",
    "700096700",
    "700096380",
    "700109173",
    "700165634",
    "700161856",
    "700023267",
    "700114125",
    "700014837",
    "700098561",
    "700035157",
    "700116028",
    "700101638",
    "700119226",
    "700038761",
    "700161742",
    "700172068",
    "700024866",
    "700161945",
    "700016960",
    "700101581",
    "700171324",
    "700015113",
    "700100227",
    "700038053",
    "700100022",
    "700033435",
    "700101916",
    "700035237",
    "700033989",
    "700107489",
    "700024449",
    "700016610",
    "700110354",
    "700095717",
    "700016210",
    "700015922",
    "700101840",
    "700117069",
    "700165569",
    "700165986",
    "700023337",
    "700037501",
    "700037123",
    "700163774",
    "700099512",
    "700110155",
    "700123959",
    "700024711",
    "700116828",
    "700165263",
    "700117755",
    "700038806",
    "700171954",
    "700095213",
    "700110089",
    "700116568",
    "700032338",
    "700033502",
    "700102356",
    "700014724",
    "700097196",
    "700116917",
    "700098429",
    "700110812",
    "700123827",
    "700095486",
    "700107759",
    "700172498",
    "700033665",
    "700101366",
    "700037868",
    "700015857",
    "700100432",
    "700033201",
    "700023113",
    "700023720",
    "700035373",
    "700107930",
    "700122000",
    "700106291",
    "700117625",
    "700164870",
    "700033153",
    "700101134",
    "700166025",
    "700106056",
    "700021902",
    "700105210",
    "700116611",
    "700034838",
    "700106198",
    "700163628",
    "700165148",
    "700106065",
    "700172221",
    "700116865",
    "700116148",
    "700095831",
    "700109506",
    "700166586",
    "700024437",
    "700106663",
    "700098669",
    "700109383",
    "700164339",
    "700171573",
    "700032413",
    "700099307",
    "700037284",
    "700117992",
    "700164686",
    "700034254",
    "700123562",
    "700111745",
    "700117682",
    "700096047",
    "700097688",
    "700117766",
    "700016542",
    "700097906",
    "700122165",
    "700165778",
    "700173023",
    "700038870",
    "700016142",
    "700102905",
    "700112433",
    "700121639",
    "700163868",
    "700015702",
    "700117938",
    "700113867",
    "700034166",
    "700109621",
    "700119987",
    "700106465",
    "700163981",
    "700038414",
    "700165809",
    "700108341",
    "700024545",
    "700037453",
    "700024752",
    "700116201",
    "700035747",
    "700107873",
    "700112376",
    "700102848",
    "700119307",
    "700034794",
    "700108530",
    "700166775",
    "700111986",
    "700106229",
    "700161532",
    "700038594",
    "700116401",
    "700024318",
    "700105372",
    "700095647",
    "700108218",
    "700032944",
    "700112046",
    "700024024",
    "700114480",
    "700024998",
    "700164159",
    "700101534",
    "700016765",
    "700100312",
    "700033797",
    "700107059",
    "700013715",
    "700097837",
    "700024615",
    "700105306",
    "700015245c",
    "700016716c",
    "700032133c",
    "700032222c",
    "700033922c",
    "700100540c",
    "700102242c",
    "700102585c",
    "700105580c",
    "700107375c",
    "700107547c",
    "700108095c",
    "700108161c",
    "700108839c",
    "700109230c",
    "700109563c",
    "700111222c",
    "700111296c",
    "700112812c",
    "700119042c",
    "700119496c",
]

# Files to use if I want to compare to Katherine's 5mers 
# List of unique HMP gut sample ids from Nandita
hmp_no_timedup_ids = [
    '700100022',
    "700113954",
    "700023788c",
    "700024509",
    "700015181",
    "700111439",
    "700163030",
    "700171066",
    "700024930",
    "700024233c",
    "700113975c",
    "700113762",
    "700117172c",
    "700102659",
    "700164050",
    "700023578",
    "700033363",
    "700015981",
    "700023919c",
    "700106170c",
    "700102299c",
    "700023845",
    "700109987",
    "700032068",
    "700116505",
    "700024673",
    "700116468",
    "700102043",
    "700171114",
    "700023634",
    "700021306",
    "700035861c",
    "700106876",
    "700171648",
    "700014562",
    "700110222",
    "700117000c",
    "700015245c",
    "700095524",
    "700116668",
    "700096865",
    "700035785c",
    "700111505",
    "700117031c",
    "700023872",
    "700111222c",
    "700037852c",
    "700021824",
    "700164450",
    "700016456c",
    "700038072",
    "700034622",
    "700111156",
    "700038386",
    "700014954",
    "700109921",
    "700099886",
    "700172726",
    "700034926c",
    "700037042c",
    "700171115",
    "700173483",
    # outlier
#    "700116730c",
    "700035533",
    "700107189c",
    "700024086",
    "700171441",
    "700015415c",
    "700021876",
    "700037738c",
    "700109449",
    # outlier
#    "700171390",
    "700034081",
    "700096700",
    "700096380c",
    "700165634",
    "700161856",
    "700111296c",
    "700023267",
    "700014837",
    "700035157",
    "700116028",
    "700101638c",
    "700038761c",
    "700024866c",
    "700016960",
    "700171324",
    "700015113c",
    "700038053",
    "700033435",
    # outlier
#    "700035237",
    "700033989c",
    "700024449",
    "700016610c",
    "700110354",
    "700095717",
    "700016210",
    "700015922",
    "700117069c",
    "700032222c",
    "700023337",
    "700037123c",
    # outlier
#    "700099512c",
    "700024711c",
    "700117755",
    "700038806c",
    "700095213c",
    "700116568",
    "700107547c",
    "700032338",
    "700033502",
    "700014724",
    "700116917",
    # outlier
#    "700098429c",
    "700095486c",
    "700172498",
    "700033665",
    "700037868",
    "700015857",
    "700033201",
    "700023113",
    "700035373c",
    "700106291c",
    "700032133c",
    "700033153c",
    "700106056",
    "700021902c",
    "700034838",
    "700106198c",
    "700106065c",
    "700116865",
    "700116148",
    "700095831c",
    "700024437",
    "700098669",
    "700164339c",
    "700032413",
    "700099307",
    "700037284c",
    "700034254c",
    "700111745",
    "700117682",
    "700096047",
    "700097688",
    "700117766",
    "700016542c",
    "700097906c",
    "700165778",
    "700173023",
    "700038870",
    "700016142c",
    "700121639",
    "700163868",
    "700015702",
    "700117938",
    "700113867",
    "700034166c",
    "700106465c",
    "700038414c",
    "700108341",
    "700024545",
    "700024752c",
    "700035747c",
    "700016716c",
    "700034794c",
    "700033922c",
    "700106229c",
    "700038594",
    "700116401",
    "700024318",
    "700095647c",
    "700032944c",
    "700024024",
    "700024998c",
    "700101534",
    "700016765",
    "700033797",
    "700013715",
    "700024615",
]

#Some files in two pieces (_1 and_2) - this will join their kmer counts into one row of a dataframe

#Setting up
path = r'/pollard/home/abustion/deep_learning_microbiome/data/jf_files/4mer_sample_tech_combined/4mer_csv_files/'
csvs = glob(os.path.join(path, "*.csv"))

one_suffix = "_1.fastq.jf.csv"
two_suffix = "_2.fastq.jf.csv"
collapsed_df = pd.DataFrame()

#Use either hmp_ids_314 or hmp_no_timedup_ids
#Load only the csvs in Nandita's list of 314 hmp samples and collapse _1 and _2 into one dataframe of counts
for csv in csvs:
    if csv.endswith(one_suffix) and str(os.path.basename(csv)).split('_')[0] in hmp_no_timedup_ids: #get the _1 that match desired 314 files
        df_1 = pd.read_csv(csv, sep = ' ', header=None).T #dataframe for _1 file
        df_1 = df_1.sort_values(axis = 1, by = [0], ascending = 1) #sort based on alphabetical order of kmers
        df_1 = df_1.drop([0]) #using .drop to remove kmers bc header was resulting in mismatch between kmer and its counts
        twin_file = str(os.path.basename(csv)).split('_')[0] + two_suffix #grab the matching _2 file
        df_2 = pd.read_csv(twin_file, sep = ' ', header=None).T #dataframe for _2 file
        df_2 = df_2.sort_values(axis = 1, by = [0], ascending = 1) #sort based kmer alphabetical order
        df_2 = df_2.drop([0])
        collapsed = df_1 + df_2
        collapsed_df = collapsed_df.append(collapsed) #add the collapsed file as one row in the final dataframe
    elif csv.endswith(two_suffix) and str(os.path.basename(csv)).split('_')[0] in hmp_no_timedup_ids:
        continue  #bc _2 already accounted for in previous block
    elif str(os.path.basename(csv)).split('.')[0] in hmp_no_timedup_ids: #for csv files without _1 or _2 suffix
        df_no_suffix = pd.read_csv(csv, sep = ' ', header=None).T
        df_no_suffix = df_no_suffix.sort_values(axis = 1, by = [0], ascending = 1)
        df_no_suffix = df_no_suffix.drop([0])
        collapsed_df = collapsed_df.append(df_no_suffix)


#header is all kmers
header = pd.read_csv('700106465_1.fastq.jf.csv', sep = ' ', header = None).T
header = header.sort_values(axis = 1, by = [0], ascending = 1)

#print(header)
#header = [
#    'AAA/TTT', 
#    'AAC/GTT', 
#    'AAG/CTT', 
#    'AAT/ATT', 
#    'ACA/TGT', 
#    'ACC/GGT', 
#    'ACG/CGT', 
#    'ACT/AGT', 
#    'AGA/TCT', 
#    'AGC/GCT', 
#    'AGG/CCT', 
#    'ATA/TAT',
#    'ATC/GAT',
#    'ATG/CAT',
#    'CAA/TTG',
#    'CAC/GTG',
#    'CAG/CTG',
#    'CCA/TGG',
#    'CCC/GGG',
#    'CCG/CGG',
#    'CGA/TCG',
#    'CGC/GCG',
#    'CTA/TAG',
#    'CTC/GAG',
#    'GAA/TTC',
#    'GAC/GTC',
#    'GCA/TGC',
#    'GCC/GGC',
#    'GGA/TCC',
#    'GTA/TAC',
#    'TAA/TTA',
#    'TCA/TGA'
#]


header.columns = header.iloc[0]
collapsed_df.columns = header.columns

#add in filenames
#Don't use for hmp_no_timedup_ids
#collapsed_df.index = hmp_ids_314

#check output
print(collapsed_df) #worked

#pickle output
collapsed_df.to_pickle('/pollard/home/abustion/deep_learning_microbiome/data/pickled_dfs/4mer_174_hmp.pickle')
