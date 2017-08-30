import mmh3
import random

'''
This program first generates 1,000,000 sequences of 33 random ATGC characters.
Secondly, it applies Murmur Hashing function on each of the sequences to obtain a
32-bit hash value.  Finally, it calculates the collision rate of the hashing function.

Output has 3 sections:

1. Hits [hash : count]
   List of all hash values and corresponding number of sequences (>1 indicates a
   collision).

2. Collisions [sequence : hash]
   List of all colliding sequences and corresponding hash values.

3. Histogram [hit : count (%)]
   Histogram of number of sequences sharing the same hash (>1 indicates a collision)
   vs corresponding number (and %) of such incidents.
'''

SEQUENCE = "ATGC"

def CreateSequence():
  return "".join(SEQUENCE[random.randint(0, 3)] for _ in range(33))

def TestInput(inputCount=100):
  print("\nWorking...")

  sequences = set([])
  hits = {}
  collisions = set([])
  first_hit = {}

  for i in range(inputCount):
    s = CreateSequence()
    while s in sequences:
      s = CreateSequence()
    sequences.add(s)
    h = mmh3.hash(s) + 2 ** 32
    hit = hits.get(h, 0)
    if hit == 0:
      first_hit[h] = s
    elif hit == 1:
      collisions.add(first_hit[h])
      collisions.add(s)
    else:
      collisions.add(s)

    hits[h] = hit + 1

  print("\nHits --------------------------------------------\n")
  histogram = {}
  for h in hits:
    hit = hits[h]
    print("%s: %d" % (h, hit))
    histogram[hit] = histogram.get(hit, 0) + 1

  print ("\nCollisions -------------------------------------\n")
  for c in collisions:
    print("%s: %d" % (c, mmh3.hash(c) + 2 ** 32))
    
  print("\nHistogram ---------------------------------------\n")
  for s in sorted(histogram):
    print("%d: %d (%3.3f%%)" % (s, histogram[s], 100.0 * s * histogram[s] / inputCount))

  print("\nDone!\n")
    
def main():
  TestInput(1000000)
  
if __name__== "__main__":
  main()
