import numpy as np
from tqdm import tqdm
from scipy.sparse import lil_matrix


def predict(matrix_U, matrix_V, topK, matrix_Train, matrix_Test, bias=None, measure="Cosine", gpu=True, is_topK=False):
    """

    :param matrix_U: embedding matrix of users
    :param matrix_V: embedding matrix of items
    :param topK: length of recommendation list
    :param matrix_Train: training feedback matrix
    :param matrix_Test: test feedback matrix
    :param bias: bias term
    :param measure: category of similarity
    :param gpu: whether to use gpu
    :return: the predicted values of all pairs in 'matrix_Test', recommendation list for user u
    """
    if gpu:
        import cupy as cp
        matrix_U = cp.array(matrix_U)
        matrix_V = cp.array(matrix_V)

    rating_predict, topK_prediction = [], []

    if is_topK:
        # for user_index in tqdm(range(matrix_U.shape[0])):
        for user_index in range(matrix_U.shape[0]):
            vector_u = matrix_U[user_index]
            vector_train = matrix_Train[user_index]
            if len(vector_train.nonzero()[0]) > 0:
                vector_predict = topk_sub_routine(vector_u, matrix_V, vector_train, bias, measure, topK=topK, gpu=gpu)
            else:
                vector_predict = np.zeros(topK, dtype=np.float32)

            topK_prediction.append(vector_predict)

    user_item_matrix = lil_matrix(matrix_Test)
    user_item_pairs = np.asarray(user_item_matrix.nonzero()).T

    rating_predict = rating_sub_routine(matrix_U, matrix_V, user_item_pairs, bias, measure, gpu=gpu)

    if is_topK:
        return rating_predict, np.vstack(topK_prediction)
    else:
        return rating_predict, topK_prediction


def topk_sub_routine(vector_u, matrix_V, vector_train, bias, measure, topK=50, gpu=True):
    """

    :param vector_u: embedding vector of user u
    :param matrix_V: embedding matrix of items
    :param vector_train: feedback vector of user u in training
    :param bias: bias term
    :param measure: category of similarity
    :param topK: length of recommendation list
    :param gpu: whether to use gpu
    :return: recommendation list for user u
    """

    train_index = vector_train.nonzero()[1]
    if measure == "Cosine":
        vector_predict = matrix_V.dot(vector_u)
    else:
        if gpu:
            import cupy as cp
            vector_predict = -cp.sum(cp.square(matrix_V - vector_u), axis=1)
        else:
            vector_predict = -np.sum(np.square(matrix_V - vector_u), axis=1)
    if bias is not None:
        if gpu:
            import cupy as cp
            vector_predict = vector_predict + cp.array(bias)
        else:
            vector_predict = vector_predict + bias

    if gpu:
        import cupy as cp
        candidate_index = cp.argpartition(-vector_predict, topK+len(train_index))[:topK+len(train_index)]
        vector_predict = candidate_index[vector_predict[candidate_index].argsort()[::-1]]
        vector_predict = cp.asnumpy(vector_predict).astype(np.float32)
    else:
        candidate_index = np.argpartition(-vector_predict, topK+len(train_index))[:topK+len(train_index)]
        vector_predict = candidate_index[vector_predict[candidate_index].argsort()[::-1]]
    vector_predict = np.delete(vector_predict, np.isin(vector_predict, train_index).nonzero()[0])

    return vector_predict[:topK]


def rating_sub_routine(matrix_U, matrix_V, user_item_pairs, bias, measure, gpu=True):
    """

    :param matrix_U: embedding matrix of users
    :param matrix_V: embedding matrix of items
    :param user_item_pairs: all (user, item) pairs in matrix_Test'
    :param bias: bias term
    :param measure: category of similarity
    :param gpu: whether to use gpu
    :return: the predicted values of all pairs in 'matrix_Test'
    """

    temp_U = matrix_U[user_item_pairs[:, 0], :]
    temp_V = matrix_V[user_item_pairs[:, 1], :]

    if measure == "Cosine":
        if gpu:
            import cupy as cp
            vector_predict = cp.sum(temp_U * temp_V, axis=1)
        else:
            vector_predict = np.sum(temp_U * temp_V, axis=1)
    else:
        if gpu:
            import cupy as cp
            vector_predict = -cp.sum(cp.square(temp_V - temp_U), axis=1)
        else:
            vector_predict = -np.sum(np.square(temp_V - temp_U), axis=1)
    if bias is not None:
        if gpu:
            import cupy as cp
            temp_bias = bias[user_item_pairs[:, 1]]
            vector_predict = vector_predict + cp.array(temp_bias)
        else:
            temp_bias = bias[user_item_pairs[:, 1]]
            vector_predict = vector_predict + temp_bias

    if gpu:
        import cupy as cp
        vector_predict = cp.asnumpy(vector_predict)
    return vector_predict
