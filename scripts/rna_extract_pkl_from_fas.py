import argparse
import os
import build_rna_features
import pickle
import random
import sys
import time
import json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def generate_pkl_from_fas(args):
    for fas in os.listdir(args.fasta_path):
        local_fasta_path = os.path.join(args.fasta_path, fas)
        feature_dir = os.path.join(args.output_dir, fas)

        timings = {}
        if args.use_precomputed_alignments is None:
            local_alignment_dir = os.path.join(feature_dir, "msa_hmm")
            if not os.path.exists(local_alignment_dir):
                os.makedirs(local_alignment_dir)
            # logging.info(f"Generating MSA for {fas} ...")
            print(f"Generating MSA for {fas} ...")
            assert "rMSA.pl" in os.listdir(args.rmsa_path), "rMSA.pl is not found. Please check that the rMSA package is installed and provide the correct path."
            cmd_perl = "perl " + os.path.join(args.rmsa_path, 'rMSA.pl ') + os.path.join(local_fasta_path, fas + '.fasta') + " -cpu={}".format(args.cpus)
            pt = time.time()
            os.system(cmd_perl)
            timings['msa'] = time.time() - pt
            if sum(1 for _ in open(os.path.join(local_fasta_path, fas + '.afa')))//2 > args.max_msa:
                cmd_head = "head -n {} {} > {}".format(args.max_msa*2, os.path.join(local_fasta_path, fas + '.afa'), os.path.join(local_alignment_dir, fas + '.afa'))
                os.system(cmd_head)
            else:
                cmd_mv = "mv " + os.path.join(local_fasta_path, fas + '.afa ') + os.path.join(local_alignment_dir, fas + '.afa')
                os.system(cmd_mv)
            cmd_mv = "mv " + os.path.join(local_fasta_path, fas + '.cm ') + os.path.join(local_alignment_dir, fas + '.cm')
            os.system(cmd_mv)
            # logging.info(f"MSA generation for {fas} is complete.")
            print(f"MSA generation for {fas} is complete.")
        else:
            local_alignment_dir = args.use_precomputed_alignments

        if args.use_precomputed_ss is None:
            local_ss_dir = os.path.join(feature_dir, "ss")
            if not os.path.exists(local_ss_dir):
                os.makedirs(local_ss_dir)
            print(f"Computing secondary structure for {fas} ...")
            cmd_pet = os.path.join(args.rmsa_path, 'bin/PETfold ')+ " -f " + os.path.join(local_alignment_dir, fas + '.afa ') + " -r " +  os.path.join(local_ss_dir, '{}_ss.txt'.format(fas))
            os.environ['PETFOLDBIN'] = os.path.join(args.rmsa_path, "data")
            pt = time.time()
            os.system(cmd_pet)
            timings['ss'] = time.time() - pt
            # logging.info(f"Secondary structure computation  for {fas} is complete.")
            print(f"Secondary structure computation for {fas} is complete.")
        else:
            local_ss_dir = args.use_precomputed_ss

        seq_file = os.path.join(local_fasta_path, '{}.fasta'.format(fas))
        msa_file = os.path.join(local_alignment_dir, '{}.afa'.format(fas))
        hmm_file = os.path.join(local_alignment_dir, '{}.cm'.format(fas))
        ss_file = os.path.join(local_ss_dir, '{}_ss.txt'.format(fas))
        features_dict = build_rna_features.processing_fas_features().collect_features(seq_file=seq_file, msa_file=msa_file, hmm_file=hmm_file, ss_file=ss_file)

        if args.add_mmcif_features is not None:
            pdbID, chainID = fas.split('_') if '_' in fas else [fas, None]
            cif_path = os.path.join(args.add_mmcif_features, pdbID + '.cif')
            assert os.path.exists(cif_path), "Cannot find file for {}.cif, Please provide the correct file location.".join(pdbID)
            features_cif = build_rna_features.processing_cif_features().collect_features(cif_path=cif_path, pdbID=pdbID, chainID=chainID, butype=features_dict['butype'])
            features_dict.update(features_cif)

        features_output_path = os.path.join(feature_dir, 'features.pkl')
        with open(features_output_path, 'wb') as f:
            pickle.dump(features_dict, f, protocol=4)
        timings_output_path = os.path.join(feature_dir, 'timings.json')
        with open(timings_output_path, 'w') as fp:
            json.dump(timings, fp, indent=4)
        # logging.info(f"process file {fas} done.")
        print(f"process file {fas} done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fasta_path", type=str,
        help="Path to directory containing FASTA files, one sequence per file"
    )
    parser.add_argument(
        "--output_dir", type=str,
    )
    parser.add_argument(
        "--use_precomputed_alignments", type=str, default=None,
        help="""Path to alignment directory. If provided, alignment computation 
                is skipped and database path arguments are ignored."""
    )
    parser.add_argument(
        "--use_precomputed_ss", type=str, default=None,
        help="""Path to secondary structure directory. If provided, secondary structure computation 
                is skipped and database path arguments are ignored."""
    )
    parser.add_argument(
        "--add_mmcif_features", type=str, default=None,
        help="""Path to the mmCIF file. If provided, features of the structure will also be added to output."""
    )
    parser.add_argument(
        "--max_msa", type=int, default=1000,
        help="""Max number of msa to use."""
    )
    parser.add_argument(
        "--cpus", type=int, default=12,
        help="""Number of CPUs with which to run alignment tools"""
    )
    parser.add_argument(
        "--rmsa_path", type=str, default="opencomplex/resources/RNA",
        help="""Path to the rMSA package. To install it to opencomplex/resources/RNA, run scripts/install_rmsa_petfold.sh"""
    )
    args = parser.parse_args()

    generate_pkl_from_fas(args)