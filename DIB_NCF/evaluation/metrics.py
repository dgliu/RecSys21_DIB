import numpy as np
from tqdm import tqdm
from scipy.sparse import lil_matrix


def recallk(vector_true_dense, hits, **unused):
    hits = len(hits.nonzero()[0])
    return float(hits)/len(vector_true_dense)


def precisionk(vector_predict, hits, **unused):
    hits = len(hits.nonzero()[0])
    return float(hits)/len(vector_predict)


def average_precisionk(vector_predict, hits, **unused):
    precisions = np.cumsum(hits, dtype=np.float32)/range(1, len(vector_predict)+1)
    return np.mean(precisions)


def r_precision(vector_true_dense, vector_predict, **unused):
    vector_predict_short = vector_predict[:len(vector_true_dense)]
    hits = len(np.isin(vector_predict_short, vector_true_dense).nonzero()[0])
    return float(hits)/len(vector_true_dense)


def _dcg_support(size):
    arr = np.arange(1, size+1)+1
    return 1./np.log2(arr)


def ndcg(vector_true_dense, vector_predict, hits):
    idcg = np.sum(_dcg_support(len(vector_true_dense)))
    dcg_base = _dcg_support(len(vector_predict))
    dcg_base[np.logical_not(hits)] = 0
    dcg = np.sum(dcg_base)
    return dcg/idcg


def click(hits, **unused):
    first_hit = next((i for i, x in enumerate(hits) if x), None)
    if first_hit is None:
        return 5
    else:
        return first_hit/10


def nll(vector_true_dense, vector_predict, **unused):
    return -1 / vector_true_dense.shape[0] * np.sum(np.log(1 + np.exp(-vector_predict * vector_true_dense)))


def auc(vector_true_dense, vector_predict, **unused):
    pos_indexes = np.where(vector_true_dense == 1)[0]
    sort_indexes = np.argsort(vector_predict)
    rank = np.nonzero(np.in1d(sort_indexes, pos_indexes))[0]
    return (
                   np.sum(rank) - len(pos_indexes) * (len(pos_indexes) + 1) / 2
           ) / (
                   len(pos_indexes) * (len(vector_predict) - len(pos_indexes))
           )


def evaluate(vector_Rating_Predict, matrix_topK_Predict, matrix_Test, metric_names, atK, analytical=False, is_topK=False):
    rating_global_metrics = {
        "NLL": nll,
        "AUC": auc
    }

    topK_global_metrics = {
        "R-Precision": r_precision,
        "NDCG": ndcg,
        "Clicks": click
    }

    topK_local_metrics = {
        "Precision": precisionk,
        "Recall": recallk,
        "MAP": average_precisionk
    }

    output = dict()

    """
    Create topK_local_metrics
    """
    if is_topK:
        num_users = matrix_topK_Predict.shape[0]

        for k in atK:

            topK_local_metric_names = list(set(metric_names).intersection(topK_local_metrics.keys()))
            results = {name: [] for name in topK_local_metric_names}
            topK_Predict = matrix_topK_Predict[:, :k]

            # for user_index in tqdm(range(topK_Predict.shape[0])):
            for user_index in range(topK_Predict.shape[0]):
                vector_predict = topK_Predict[user_index]
                if len(vector_predict.nonzero()[0]) > 0:
                    vector_true = matrix_Test[user_index]
                    vector_true_dense = (vector_true > 0).nonzero()[1]
                    hits = np.isin(vector_predict, vector_true_dense)

                    if vector_true_dense.size > 0:
                        for name in topK_local_metric_names:
                            results[name].append(topK_local_metrics[name](vector_true_dense=vector_true_dense,
                                                                          vector_predict=vector_predict,
                                                                          hits=hits))

            results_summary = dict()
            if analytical:
                for name in topK_local_metric_names:
                    results_summary['{0}@{1}'.format(name, k)] = results[name]
            else:
                for name in topK_local_metric_names:
                    results_summary['{0}@{1}'.format(name, k)] = (np.average(results[name]),
                                                                  1.96*np.std(results[name])/np.sqrt(num_users))
            output.update(results_summary)

        """
        Create topK_global_metrics
        """
        topK_global_metric_names = list(set(metric_names).intersection(topK_global_metrics.keys()))
        results = {name: [] for name in topK_global_metric_names}

        topK_Predict = matrix_topK_Predict[:]

        # for user_index in tqdm(range(topK_Predict.shape[0])):
        for user_index in range(topK_Predict.shape[0]):
            vector_predict = topK_Predict[user_index]

            if len(vector_predict.nonzero()[0]) > 0:
                vector_true = matrix_Test[user_index]
                vector_true_dense = (vector_true > 0).nonzero()[1]
                hits = np.isin(vector_predict, vector_true_dense)

                # if user_index == 1:
                #     import ipdb;
                #     ipdb.set_trace()

                if vector_true_dense.size > 0:
                    for name in topK_global_metric_names:
                        results[name].append(topK_global_metrics[name](vector_true_dense=vector_true_dense,
                                                                       vector_predict=vector_predict,
                                                                       hits=hits))

        results_summary = dict()
        if analytical:
            for name in topK_global_metric_names:
                results_summary[name] = results[name]
        else:
            for name in topK_global_metric_names:
                results_summary[name] = (np.average(results[name]), 1.96*np.std(results[name])/np.sqrt(num_users))
        output.update(results_summary)

    """
    Create rating_global_metrics
    """
    user_item_matrix = lil_matrix(matrix_Test)
    user_item_pairs = np.asarray(user_item_matrix.nonzero()).T
    vector_true_dense = np.asarray(matrix_Test[user_item_pairs[:, 0], user_item_pairs[:, 1]])[0]
    vector_true_dense[vector_true_dense == -1] = 0

    rating_global_metric_names = list(set(metric_names).intersection(rating_global_metrics.keys()))
    results = {name: [] for name in rating_global_metric_names}

    for name in rating_global_metric_names:
        results[name].append(rating_global_metrics[name](vector_true_dense=vector_true_dense,
                                                         vector_predict=vector_Rating_Predict))

    results_summary = dict()
    for name in rating_global_metric_names:
        results_summary[name] = (np.average(results[name]), np.float64(0))
    output.update(results_summary)

    return output