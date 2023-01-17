import argparse
import os

import pickle
import random
import sys
import time
import json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from opencomplex.data import templates, data_pipeline


def generate_pkl_from_fas(args):        
    template_featurizer = templates.TemplateHitFeaturizer(
        mmcif_dir=args.template_mmcif_dir,
        max_template_date=args.max_template_date,
        max_hits=args.max_templates,
        kalign_binary_path=args.kalign_binary_path,
        release_dates_path=args.release_dates_path,
        obsolete_pdbs_path=args.obsolete_pdbs_path
    )

    data_processor = data_pipeline.DataPipeline(
        template_featurizer=template_featurizer,
    )
    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(sys.maxsize)
    
    alignment_runner = data_pipeline.AlignmentRunner(
        jackhmmer_binary_path=args.jackhmmer_binary_path,
        hhblits_binary_path=args.hhblits_binary_path,
        hhsearch_binary_path=args.hhsearch_binary_path,
        uniref90_database_path=args.uniref90_database_path,
        mgnify_database_path=args.mgnify_database_path,
        bfd_database_path=args.bfd_database_path,
        uniclust30_database_path=args.uniclust30_database_path,
        pdb70_database_path=args.pdb70_database_path,
        use_small_bfd=args.use_small_bfd,
        no_cpus=args.cpus,
    )
    for fas in os.listdir(args.fasta_path):
        local_fasta_path=os.path.join(args.fasta_path,fas)
        fas_name=fas.split('.')[0]
        feature_dir=os.path.join(args.output_dir, fas_name)
    
        if args.use_precomputed_alignments is None:
            local_alignment_dir = os.path.join(feature_dir, "msas")
            if not os.path.exists(local_alignment_dir):
                os.makedirs(local_alignment_dir)
        else:
            local_alignment_dir = args.use_precomputed_alignments
        

        # logging.info(f"Generating features for {fas_name} ...") 
        print(f"Generating features for {fas_name} ...")
        # if timings is None:
        timings = {}
        pt = time.time()
        alignment_runner.run(
            local_fasta_path, local_alignment_dir
        )    
        feature_dict = data_processor.process_fasta(
            fasta_path=local_fasta_path, alignment_dir=local_alignment_dir
        )
        timings['data_pipeline'] = time.time() - pt
        features_output_path = os.path.join(feature_dir, 'features.pkl')
        with open(features_output_path, 'wb') as f:
            pickle.dump(feature_dict, f, protocol=4)
        timings_output_path = os.path.join(feature_dir, 'timings.json')
        with open(timings_output_path, 'w') as fp:
            json.dump(timings, fp, indent=4)
        # logging.info(f"process file {fas_name} done.")
        print(f"process file {fas_name} done.")
        
         
if __name__ == "__main__":
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
        "--max_templates", type=int, default=20,
        help="""Max number of templates to use."""
    )
    parser.add_argument(
        "--cpus", type=int, default=12,
        help="""Number of CPUs with which to run alignment tools"""
    )
    parser.add_argument(
        "--data_random_seed", type=str, default=None
    )
    parser.add_argument(
        "--template_mmcif_dir", type=str, default="/mnt/database/pdb_mmcif/mmcif_files",
        help="""Path to mmcif directory."""
    )
    parser.add_argument(
        "--uniref90_database_path", type=str, default="/mnt/database/uniref90_latest/uniref90.fasta",
        help="""Path to uniref90 directory."""
    )
    parser.add_argument(
        "--mgnify_database_path", type=str, default="/mnt/database/mgnify/mgy_clusters.fa",
        help="""Path to mgnify directory."""
    )
    parser.add_argument(
        "--pdb70_database_path", type=str, default="/mnt/database/pdb70_latest/pdb70",
        help="""Path to pdb70 directory."""
    )
    parser.add_argument(
        "--uniclust30_database_path", type=str, default="/mnt/database/uniref30_latest/UniRef30_2021_03",
        help="""Path to uniclust30 directory."""
    )
    parser.add_argument(
        "--bfd_database_path", type=str, default="/mnt/database/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt",
        help="""Path to bfd directory."""
    )
    parser.add_argument(
        "--use_small_bfd", default=False, action="store_true",
        help="""If use small_bfd."""
    )
    parser.add_argument(
        "--obsolete_pdbs_path", type=str, default="/mnt/database/pdb_mmcif/obsolete.dat",
        help="""Path to obsolete_pdbs_path ."""
    )
    parser.add_argument(
        "--jackhmmer_binary_path", type=str, default="/opt/conda/envs/opencomplex_venv/bin/jackhmmer",
        help="""Binary path of jackhmmer."""
    )
    parser.add_argument(
        "--hhblits_binary_path", type=str, default="/opt/cnda/envs/opencomplex_venv/bin/hhblits",
        help="""Binary path of hhblits."""
    )
    parser.add_argument(
        "--hhsearch_binary_path", type=str, default="/opt/conda/envs/opencomplex_venv/bin/hhsearch",
        help="""Binary path of hhsearch."""
    )
    parser.add_argument(
        "--kalign_binary_path", type=str, default="/opt/conda/envs/opencomplex_venv/bin/kalign",
        help="""Binary path of kalign."""
    )
    parser.add_argument(
        "--max_template_date", type=str, default="2022-04-24",
        help="""max_template_date."""
    )
    parser.add_argument(
        "--release_dates_path", type=str, default=None,
        help="""release_dates_path."""
    )
    args = parser.parse_args()

    generate_pkl_from_fas(args)