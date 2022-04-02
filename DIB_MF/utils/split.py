import numpy as np
from tqdm import tqdm
import scipy.sparse as sparse
from scipy.sparse import lil_matrix


def split_seed_randomly(rating_matrix, ratio, threshold, implicit, remove_empty, split_seed):
    """
    Split based on a deterministic seed randomly
    """
    if implicit:
        '''
        If only implicit (clicks, views, binary) feedback, convert to implicit feedback
        '''
        temp_rating_matrix = sparse.csr_matrix(rating_matrix.shape)
        temp_rating_matrix[rating_matrix.nonzero()] = -1
        temp_rating_matrix[(rating_matrix >= threshold)] = 1
        rating_matrix = temp_rating_matrix

    if remove_empty:
        # Remove empty columns. record original item index
        nonzero_index = np.unique(rating_matrix.nonzero()[1])
        rating_matrix = rating_matrix[:, nonzero_index]

        # Remove empty rows. record original user index
        nonzero_rows = np.unique(rating_matrix.nonzero()[0])
        rating_matrix = rating_matrix[nonzero_rows]

    # Set the random seed for splitting
    np.random.seed(split_seed)

    # Randomly shuffle the data
    user_item_matrix = lil_matrix(rating_matrix)
    user_item_pairs = np.asarray(user_item_matrix.nonzero()).T

    index_shuf = np.arange(len(user_item_pairs))
    np.random.shuffle(index_shuf)
    user_item_pairs = user_item_pairs[index_shuf]

    rows, cols, rating = user_item_pairs[:, 0], user_item_pairs[:, 1], np.asarray(
        rating_matrix[user_item_pairs[:, 0], user_item_pairs[:, 1]]).flatten()
    num_nonzeros = len(rows)

    # Convert to csr matrix
    cut_point = int(ratio[0] * num_nonzeros)

    train = sparse.csr_matrix((rating[:cut_point], (rows[:cut_point], cols[:cut_point])),
                              shape=rating_matrix.shape, dtype=np.int32)

    valid = sparse.csr_matrix((rating[cut_point:], (rows[cut_point:], cols[cut_point:])),
                              shape=rating_matrix.shape, dtype=np.int32)

    return train, valid


def split_seed_randomly_by_user(rating_matrix, ratio, threshold, implicit, remove_empty, split_seed):
    """
    Split based on a deterministic seed randomly and group by user
    """
    if implicit:
        '''
        If only implicit (clicks, views, binary) feedback, convert to implicit feedback
        '''
        temp_rating_matrix = sparse.csr_matrix(rating_matrix.shape)
        temp_rating_matrix[rating_matrix.nonzero()] = -1
        temp_rating_matrix[(rating_matrix >= threshold)] = 1
        rating_matrix = temp_rating_matrix

    if remove_empty:
        # Remove empty columns. record original item index
        nonzero_index = np.unique(rating_matrix.nonzero()[1])
        rating_matrix = rating_matrix[:, nonzero_index]

        # Remove empty rows. record original user index
        nonzero_rows = np.unique(rating_matrix.nonzero()[0])
        rating_matrix = rating_matrix[nonzero_rows]

    # Note: This just gets the highest userId and doesn't account for non-contiguous users.
    user_num, item_num = rating_matrix.shape

    train = []
    valid = []

    # Set the random seed for splitting
    np.random.seed(split_seed)
    for i in tqdm(range(user_num)):
        item_indexes = rating_matrix[i].nonzero()[1]
        rating_data = rating_matrix[i].data
        num_nonzeros = len(item_indexes)
        if num_nonzeros >= 1:
            num_valid = int(num_nonzeros * ratio[1])

            valid_offset = num_nonzeros - num_valid

            # Randomly shuffle the data
            permuteIndices = np.random.permutation(rating_data.size)
            rating_data = rating_data[permuteIndices]
            item_indexes = item_indexes[permuteIndices]

            # Append data to train, valid, test
            train.append([rating_data[:valid_offset], np.full(valid_offset, i), item_indexes[:valid_offset]])
            valid.append([rating_data[valid_offset:], np.full(num_valid, i), item_indexes[valid_offset:]])

    # Convert to csr matrix
    train = np.array(train)
    valid = np.array(valid)

    train = sparse.csr_matrix((np.hstack(train[:, 0]), (np.hstack(train[:, 1]), np.hstack(train[:, 2]))),
                              shape=rating_matrix.shape, dtype=np.int32)
    valid = sparse.csr_matrix((np.hstack(valid[:, 0]), (np.hstack(valid[:, 1]), np.hstack(valid[:, 2]))),
                              shape=rating_matrix.shape, dtype=np.int32)

    return train, valid
