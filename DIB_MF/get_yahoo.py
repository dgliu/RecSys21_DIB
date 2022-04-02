import os
import argparse
import numpy as np
from scipy.sparse import csr_matrix
from utils.argcheck import ratio_without_test
from utils.progress import WorkSplitter
from utils.io import save_numpy, load_pandas_without_names
from utils.split import split_seed_randomly, split_seed_randomly_by_user


def main(args):
    progress = WorkSplitter()

    save_dir = args.path + args.problem
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    progress.section("Yahoo R3: Load Original Data")

    user_rating_matrix = load_pandas_without_names(path=args.path, name=args.problem + args.user, sep=args.sep,
                                                   df_name=args.df_name, value_name='rating')
    test_mat = load_pandas_without_names(path=args.path, name=args.problem + args.random, sep=args.sep,
                                         df_name=args.df_name, value_name='rating',
                                         shape=user_rating_matrix.shape)

    progress.section("Yahoo R3: Split Original Data")

    if args.split == 'user':
        train_mat, valid_mat = split_seed_randomly_by_user(rating_matrix=user_rating_matrix,
                                                           ratio=args.ratio,
                                                           threshold=args.threshold,
                                                           implicit=args.implicit,
                                                           remove_empty=args.remove_empty,
                                                           split_seed=args.seed)

    else:
        train_mat, valid_mat = split_seed_randomly(rating_matrix=user_rating_matrix,
                                                   ratio=args.ratio,
                                                   threshold=args.threshold,
                                                   implicit=args.implicit,
                                                   remove_empty=args.remove_empty,
                                                   split_seed=args.seed)

    save_numpy(train_mat, save_dir, "train_" + args.split)
    save_numpy(valid_mat, save_dir, "valid_" + args.split)

    if args.implicit:
        '''
        If only implicit (clicks, views, binary) feedback, convert to implicit feedback
        '''
        temp_rating_matrix = csr_matrix(test_mat.shape, dtype=np.int32)
        temp_rating_matrix[test_mat.nonzero()] = -1
        temp_rating_matrix[(test_mat >= args.threshold)] = 1
        test_mat = temp_rating_matrix
    save_numpy(test_mat, save_dir, "test")

    progress.section("Data Preprocess Completed")


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Data Preprocess")
    parser.add_argument('-p', dest='path', default='datax/')
    parser.add_argument('-d', dest='problem', default='yahooR3/')
    parser.add_argument('-user', dest='user', help='user subset', default='user.txt')
    parser.add_argument('-random', dest='random', help='random subset', default='random.txt')
    parser.add_argument('-sep', dest='sep', help='separate', default=',')
    parser.add_argument('-dn', dest='df_name', help='column names of dataframe',
                        default=['uid', 'iid', 'rating'])
    parser.add_argument('-s', dest='seed', help='random seed', type=int, default=0)
    parser.add_argument('-sp', dest='split', help='split mode', default='user')  # or 'random'
    parser.add_argument('-r', dest='ratio', type=ratio_without_test, default='0.8,0.2')
    parser.add_argument('-threshold', dest='threshold', default=4)
    parser.add_argument('--implicit', dest='implicit', action='store_false', default=True)
    parser.add_argument('--remove_empty', dest='remove_empty', action='store_false', default=True)
    args = parser.parse_args()
    main(args)
