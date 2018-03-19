library("Rtsne")
library(foreach)
library(data.table)

###################
## SPCS RELATIVE ABUNDANCE
###################

## read in relative abundance table
relAb=read.table(file="relative_abundance.txt",header=1,row.names=1)
# dim(relAb) ---> [5952 1573] --> 5952 spcs, 1573 samples

## read in sample info
sampInfo=read.table(file="metafiles/HMP_ids.txt",sep="\t",header=1)
sampInfo.noRelAbRow=sampInfo[!is.element(sampInfo$run_accession,colnames(relAb)),]
sampInfo=sampInfo[match(colnames(relAb),sampInfo$run_accession),]

## remove rows for species not represented in any sample 
spcsSums=rowSums(relAb)
relAb=relAb[spcsSums>0,]
# dim(relAb) ---> [1133 1573] --> 1133 present spcs, 1573 samples

## remove cols for samples without any spcs represented (?? why)
sampSums=colSums(relAb)
sampInfo.null=sampInfo[sampSums==0,]
sampInfo=sampInfo[sampSums>0,]
relAb=relAb[,sampSums>0]
# dim(relAb) ---> [1133 1569] --> 1133 present spcs, 1569 unique samples,

## run tsne
tsne_out <- Rtsne(t(relAb))
# dim(tsne_out$Y) ---> [1569 2]

## write
relAb.tsne=cbind(sampInfo,relAb.tsne_out$Y)
colnames(relAb.tsne)[6:7]=c("x","y")
write.csv(relAb.tsne,file="relative_abundance.tsne_out.csv",quote=F,row.names = F)

## plot
##   use same color for samps from same subjects; color singleton samps black
subjectIDs=table(relAb.tsne$subject_id)
plotColors=c("black",rainbow(length(subjectIDs)))
subjectIX=match(relAb.tsne$subject_id,rownames(subjectIDs))
subjectIX[subjectIDs[subjectIX]==1]=0
plot(relAb.tsne$x,relAb.tsne$y,col=plotColors[1+subjectIX])

###################
## 5-MERS
###################
## read in 2161 kmer count files
fl=system("ls 5mer_cts/*.rawcnt",intern=T)
kmerCts=foreach(ii=fl,.combine=rbind)%do%{a=scan(ii)} 
# dim(kmerCts) ---> [2161 1024] --> 2161 samples, 1024 kmers

## read in metadata
mdFiles=paste0("metafiles/",c("arthritis_metaData_merged.txt",
    "HMP_ids.txt",
    "MetaHIT_ids.txt",
    "Qin_2012_ids_all.txt"
))
sampInfo=rbindlist(foreach(ff=mdFiles)%do%{read.table(ff,header=1,sep="\t")}, use.names=TRUE, fill=TRUE)
sampInfo$country[!is.na(sampInfo$arthritis)]="China"
## match metadata to kmer matrix rows
fl.sampID=sapply(strsplit(sapply(strsplit(basename(fl),"\\."),"[",1),"_"),"[",1)
ix=match(fl.sampID,sampInfo$run_accession)
ix[is.na(ix)]=match(fl.sampID[is.na(ix)],sampInfo$sample_id)
sampInfo=sampInfo[ix,]

## any null/duplicate samples?
nrow(kmerCts)==nrow(unique(kmerCts))  #(? --> FALSE)
# get indices of dup samps
dupIX=c(which(duplicated(kmerCts)),which(duplicated(kmerCts,fromLast=TRUE)))
## weird - did we know this? they are from the same ind but diff samp IDs. picking one randomly
kmerCts=kmerCts[-dupIX[2],]
sampInfo=sampInfo[-dupIX[2],]

## any null/duplicate kmers?
ncol(kmerCts)==nrow(unique(t(kmerCts)))  #(? --> TRUE)

## run tsne
tsne_out <- Rtsne(kmerCts,perplexity = 30)
# dim(tsne_out$Y) ---> [1569 2]

## write
kmerCts.tsne=cbind(sampInfo,tsne_out$Y)
colnames(kmerCts.tsne)[15:16]=c("tsne.x","tsne.y")
write.csv(kmerCts.tsne,file="5mer_cts.tsne_out.csv",quote=F,row.names = F)

## plot
##   use same color for samps from same subjects; color singleton samps black
countries=unique(kmerCts.tsne$country)
plotColors=rainbow(length(countries))
countryIX=match(kmerCts.tsne$country,countries)
disease=rep(0,nrow(kmerCts.tsne))
disease[kmerCts.tsne$arthritis==1]=1
disease[kmerCts.tsne$ibd==1]=2
disease[kmerCts.tsne$bmi_type=="obese"]=3
disease[kmerCts.tsne$t2d==1]=4

pdf(file="5mer_cts.tsne_scatter.pdf",height=11,width=13)
plot(kmerCts.tsne$tsne.x,kmerCts.tsne$tsne.y,col=plotColors[countryIX],pch=1+disease)
legend(15,-30,legend=countries,col=plotColors,pch=1,cex=.6,bty="n")
legend(30,-30,legend=c("arthritis","ibd","obesity","diabetes"),pch=1:4,cex=.6,bty="n")
dev.off()
