from sklearn.preprocessing import normalize

# Different ways of normalizing each sample's kmer count vector
def norm_by_l1(kmers):
    # normalize along the second dimension of the matrix, i.e., per row
    return normalize(kmers, axis=1, norm='l1')

def norm_by_l2(kmers):
    # normalize along the second dimension of the matrix, i.e., per row
    return normalize(kmers, axis=1, norm='l2')

