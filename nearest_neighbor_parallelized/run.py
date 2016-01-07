import sys

import nearest_neighbor_parallelized

if __name__ == "__main__":
    path = sys.argv[1] 
    nearest_neighbor_parallelized.classify(path)
