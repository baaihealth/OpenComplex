import argparse
import multiprocessing as mp

from functools import partial
from tqdm import tqdm

from opencomplex.utils import metric_tool


def main(target_file, args):
    metric_tool.compute_metric(
        args.native_dir,
        target_file,
        mode="multimer" if args.multimer else "monomer",
        complex_type=args.complex_type,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction_dir", type=str,
        help="""Directory to predicted pdb files."""
    )
    parser.add_argument(
        "--native_dir", type=str,
        help="""Directory to mmcif files."""
    )
    parser.add_argument(
        "--target_list_file", type=str,
        help="""File path to target list."""
    )
    parser.add_argument(
        "--complex_type", type=str,
        default="protein", choices=["protein", "RNA", "mix"],
        help="""Complex type of predictions."""
    )
    parser.add_argument(
        "--multimer", action="store_true",
        help="""If the prediction has multiple chains."""
    )
    parser.add_argument(
        "--num_workers", type=int, default=8,
        help="""Number of workers to compute metrics in parallel."""
    )

    args = parser.parse_args()


    
    target_list = metric_tool.get_prediction_list(args.prediction_dir, args.target_list_file)
    worker = partial(main, args=args)
    with mp.Pool(args.num_workers) as p:
        list(tqdm(p.imap_unordered(worker, target_list), total=len(target_list)))
    p.join()

    metric_tool.summarize_metrics(args.prediction_dir, target_list_file=args.target_list_file)