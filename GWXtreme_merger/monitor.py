import os
import time
import argparse

def monitor(dir_length):

    while True:
        results = os.listdir("data/multiprocessing/")
        time.sleep(1)
        if len(results) == (dir_length):
            break

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_length', help="Number of files in directory", type=int)
    args = parser.parse_args()

    monitor(args.dir_length)

