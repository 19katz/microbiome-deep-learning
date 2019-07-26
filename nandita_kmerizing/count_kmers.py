#!/usr/bin/env python3
#~/miniconda3/bin/python3


import gzip
import pandas as pd
import numpy as np
import sys
from glob import glob
import os.path

inFN=sys.argv[1]
inFile=gzip.open(inFN, 'rb')

total_count=0
for line in inFile: 
    line=line.strip('\n')
    value=int(line[1:])
    total_count +=value


print inFN + '\t' + str(total_count)
