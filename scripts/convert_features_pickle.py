#!env python

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
import random
import pickle
import numpy as np
from multiprocessing import Process

from opencomplex.data import mmcif_parsing

type_map = {'butype': np.uint8,
        'between_segment_residues': np.uint8,
        'deletion_matrix_int': np.uint16,
        'msa': np.uint8,
        'template_butype': np.uint8,
        'template_all_atom_masks': np.uint8,
        }

def convert_dtype(fea):
    for k in type_map:
        fea[k] = fea[k].astype(type_map[k])
    return fea

def convert_pickle(input_file, output_path):
    with open(input_file, 'rb') as fp:
        fea = pickle.load(fp)
    fea = convert_dtype(fea)
    with open(os.path.join(output_path, "features.pkl"), 'wb') as fp:
        pickle.dump(fea, fp)

def preprocess_mmcif(file_id, input_file, output_path, min_process_time_to_write=1.0):
    with open(input_file, 'r') as f:
        mmcif_string = f.read()

    t0 = time.time()
    mmcif_object = mmcif_parsing.parse(
            file_id=file_id, mmcif_string=mmcif_string
            )
    t1 = time.time()

    if (mmcif_object.mmcif_object is None or t1 - t0 <= min_process_time_to_write):
        return

    mmcif_object = mmcif_object.mmcif_object
    with open(os.path.join(output_path, "mmcif_obj.pkl"), 'wb') as fp:
        pickle.dump(mmcif_object, fp)
    print("Wrote mmcif object")

class FeaProcessor(Process):
    def __init__(self, flist, raw_feature_path, raw_mmcif_path, new_feature_path, min_process_time_to_write):
        super(FeaProcessor, self).__init__()
        self.flist = flist
        random.shuffle(self.flist)
        self.raw_feature_path = raw_feature_path
        self.raw_mmcif_path = raw_mmcif_path
        self.new_feature_path = new_feature_path
        self.min_process_time_to_write = min_process_time_to_write

    def run(self):
        for name in self.flist:
            output_path = os.path.join(self.new_feature_path, name)
            if os.path.exists(output_path):
                continue
            os.mkdir(output_path)

            print(f"Processing {name}")
            spl = name.rsplit('_', 1)
            if(len(spl) == 2):
                file_id, chain_id = spl
            else:
                file_id, = spl
                chain_id = None

            cif_path = os.path.join(self.raw_mmcif_path, file_id) + ".cif"
            feature_path = os.path.join(self.raw_feature_path, name, "features.pkl")

            convert_pickle(feature_path, output_path)
            preprocess_mmcif(file_id, cif_path, output_path, min_process_time_to_write=self.min_process_time_to_write)


if __name__ == "__main__":
    raw_feature_path = "/sharefs/baaihealth/public_datasets/CASP/dataset0225/train/features"
    raw_mmcif_path = "/sharefs/baaihealth/public_datasets/CASP/dataset0225/train/mmcif"
    new_feature_path = "/sharefs/baaihealth/public_datasets/CASP/dataset0225/train/new_features"
    # raw_feature_path = "/sharefs/baaihealth/public_datasets/CASP/dataset0329/dataset0329/features"
    # raw_mmcif_path = "/sharefs/baaihealth/public_datasets/CASP/dataset0329/dataset0329/mmcif"
    # new_feature_path = "/sharefs/baaihealth/public_datasets/CASP/dataset0329/dataset0329/new_features"

    min_process_time_to_write = 1.0
    process_no = 40

    flist = os.listdir(raw_feature_path)

    pros = []
    for i in range(process_no):
        p = FeaProcessor(flist, raw_feature_path, raw_mmcif_path, new_feature_path, min_process_time_to_write)
        p.start()
        pros.append(p)

    for i in range(len(pros)):
        pros[i].join()

