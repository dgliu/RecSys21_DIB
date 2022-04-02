import numpy as np
from tqdm import tqdm
import tensorflow as tf
import optuna
from optuna.samplers import TPESampler
from optuna.trial import Trial
from evaluation.metrics import evaluate
from utils.progress import WorkSplitter
from utils.regularizers import Regularizer
from scipy.sparse import vstack, lil_matrix, csr_matrix


class Objective:

    def __init__(self, num_users, num_items, optimizer, gpu_on, train, valid, test, iters, metric, is_topK,
                 topK, seed) -> None:
        """Initialize Class"""
        self.num_users = num_users
        self.num_items = num_items
        self.optimizer = optimizer
        self.gpu_on = gpu_on
        self.train = train
        self.valid = valid
        self.test = test
        self.iters = iters
        self.metric = metric
        self.is_topK = is_topK
        self.topK = topK
        self.seed = seed

    def __call__(self, trial: Trial) -> float:
        """Calculate an objective value."""

        # sample a set of hyperparameters.
        rank = trial.suggest_discrete_uniform('rank', 5, 100, 5)
        lam = trial.suggest_categorical('lambda', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
        batch_size = trial.suggest_categorical('batch_size', [128, 256, 512, 1024, 2048])
        lr = trial.suggest_categorical('learning_rate', [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05])
        alpha = trial.suggest_uniform('alpha', 0.1, 0.2)
        gamma = trial.suggest_uniform('gamma', 0.001, 0.005)

        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        model = DIBNCF(self.num_users, self.num_items, np.int(rank), np.int(batch_size), lamb=lam, alpha=alpha,
                       gamma=gamma, learning_rate=lr, optimizer=self.optimizer, gpu_on=self.gpu_on)

        score, _, _, _ = model.train_model(self.train, self.valid, self.test, self.iters, self.metric, self.topK,
                                           self.is_topK, self.gpu_on, self.seed)

        model.sess.close()
        tf.reset_default_graph()

        return score


class Tuner:
    """Class for tuning hyperparameter of NCF models."""

    def __init__(self):
        """Initialize Class."""

    def tune(self, n_trials, num_users, num_items, optimizer, gpu_on, train, valid, test, epoch, metric, topK,
             is_topK, seed):
        """Hyperparameter Tuning by TPE."""
        objective = Objective(num_users=num_users, num_items=num_items, optimizer=optimizer, gpu_on=gpu_on, train=train,
                              valid=valid, test=test, iters=epoch, metric=metric, is_topK=is_topK, topK=topK, seed=seed)
        study = optuna.create_study(sampler=TPESampler(seed=seed), direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        return study.trials_dataframe(), study.best_params


class DIBNCF(object):
    def __init__(self, num_users, num_items, embed_dim, batch_size,
                 lamb=0.01,
                 alpha=0.01,
                 gamma=0.01,
                 learning_rate=1e-3,
                 optimizer=tf.train.AdamOptimizer,
                 gpu_on=False,
                 **unused):
        self._num_users = num_users
        self._num_items = num_items
        self._embed_dim = embed_dim
        self._lamb = lamb
        self._alpha = alpha
        self._gamma = gamma
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._optimizer = optimizer
        self._gpu_on = gpu_on
        self._build_graph()

    def _build_graph(self):

        with tf.variable_scope('dib-ncf'):
            # Placehoder
            self.user_idx = tf.placeholder(tf.int32, [None])
            self.item_idx = tf.placeholder(tf.int32, [None])
            self.label = tf.placeholder(tf.float32, [None])

            # Variable to learn
            z_user_embeddings = tf.Variable(tf.random_normal([self._num_users, self._embed_dim],
                                                             stddev=1 / (self._embed_dim ** 0.5), dtype=tf.float32))
            c_user_embeddings = tf.Variable(tf.random_normal([self._num_users, self._embed_dim],
                                                             stddev=1 / (self._embed_dim ** 0.5), dtype=tf.float32))

            user_zero_vector = tf.get_variable(
                'user_zero_vector', [self._num_users, self._embed_dim],
                initializer=tf.constant_initializer(0.0, dtype=tf.float32), trainable=False)
            self.z_user_embeddings = tf.concat([z_user_embeddings, user_zero_vector], 1)
            self.c_user_embeddings = tf.concat([user_zero_vector, c_user_embeddings], 1)

            z_item_embeddings = tf.Variable(tf.random_normal([self._num_items, self._embed_dim],
                                                             stddev=1 / (self._embed_dim ** 0.5), dtype=tf.float32))
            c_item_embeddings = tf.Variable(tf.random_normal([self._num_items, self._embed_dim],
                                                             stddev=1 / (self._embed_dim ** 0.5), dtype=tf.float32))

            item_zero_vector = tf.get_variable(
                'item_zero_vector', [self._num_items, self._embed_dim],
                initializer=tf.constant_initializer(0.0, dtype=tf.float32), trainable=False)

            self.z_item_embeddings = tf.concat([z_item_embeddings, item_zero_vector], 1)
            self.c_item_embeddings = tf.concat([item_zero_vector, c_item_embeddings], 1)

            self.mlp1_weights = tf.Variable(tf.truncated_normal([self._embed_dim * 4, self._embed_dim * 2],
                                                                stddev=1 / 500.0), name="layer1_weights")
            self.mlp1_bias = tf.Variable(tf.constant(0., shape=[self._embed_dim * 2]), name="layer1_bias")

            self.mlp2_weights = tf.Variable(tf.truncated_normal([self._embed_dim * 2, 1],
                                                                stddev=1 / 500.0), name="layer2_weights")

            with tf.variable_scope("ncf_loss"):
                z_users = tf.nn.embedding_lookup(self.z_user_embeddings, self.user_idx)
                z_items = tf.nn.embedding_lookup(self.z_item_embeddings, self.item_idx)

                z = tf.concat([z_users, z_items], axis=1)

                z_encoded = tf.matmul(z, self.mlp1_weights) + self.mlp1_bias
                z_x_ij = tf.squeeze(tf.matmul(tf.nn.tanh(z_encoded), self.mlp2_weights))
                self.sigmoid_z_x_ij = tf.nn.sigmoid(z_x_ij)

                c_users = tf.nn.embedding_lookup(self.c_user_embeddings, self.user_idx)
                c_items = tf.nn.embedding_lookup(self.c_item_embeddings, self.item_idx)

                c = tf.concat([c_users, c_items], axis=1)

                c_encoded = tf.matmul(c, self.mlp1_weights) + self.mlp1_bias
                c_x_ij = tf.squeeze(tf.matmul(tf.nn.tanh(c_encoded), self.mlp2_weights))

                zc_users = z_users + c_users
                zc_items = z_items + c_items
                zc = tf.concat([zc_users, zc_items], axis=1)

                zc_encoded = tf.matmul(zc, self.mlp1_weights) + self.mlp1_bias
                zc_x_ij = tf.squeeze(tf.matmul(tf.nn.tanh(zc_encoded), self.mlp2_weights))

                mf_loss = tf.reduce_mean(
                    (1 - self._alpha) * tf.nn.sigmoid_cross_entropy_with_logits(logits=z_x_ij, labels=self.label) -
                    self._gamma * tf.nn.sigmoid_cross_entropy_with_logits(logits=c_x_ij, labels=self.label) +
                    self._alpha * tf.nn.sigmoid_cross_entropy_with_logits(logits=zc_x_ij, labels=self.label))

                self.a = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=z_x_ij, labels=self.label))
                self.b = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=c_x_ij, labels=self.label))
                self.c = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=zc_x_ij, labels=self.label))

            with tf.variable_scope('l2_loss'):
                unique_user_idx, _ = tf.unique(self.user_idx)
                unique_users = tf.nn.embedding_lookup(self.z_user_embeddings, unique_user_idx)

                unique_item_idx, _ = tf.unique(self.item_idx)
                unique_items = tf.nn.embedding_lookup(self.z_item_embeddings, unique_item_idx)

                l2_loss = tf.reduce_mean(
                    tf.nn.l2_loss(unique_users)) + tf.reduce_mean(
                    tf.nn.l2_loss(unique_items)) + tf.reduce_mean(
                    tf.nn.l2_loss(self.mlp1_weights)) + tf.reduce_mean(
                    tf.nn.l2_loss(self.mlp2_weights))

            with tf.variable_scope('loss'):
                self._loss = mf_loss + self._lamb * l2_loss

            with tf.variable_scope('optimizer'):
                optimizer = self._optimizer(learning_rate=self._learning_rate)

            with tf.variable_scope('training-step'):
                self._train = optimizer.minimize(self._loss)

            with tf.variable_scope('top-K'):
                self.prediction_top_k = tf.placeholder(tf.float32, [None, None])
                self.scale_top_k = tf.placeholder(tf.int32)
                self.top_k = tf.nn.top_k(self.prediction_top_k, self.scale_top_k)

            if self._gpu_on:
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
            else:
                config = tf.ConfigProto(device_count={'GPU': 0})
            self.sess = tf.Session(config=config)
            init = tf.global_variables_initializer()
            self.sess.run(init)

    @staticmethod
    def get_batches(user_item_pairs, rating_matrix, unlabeled_ui, num_unlabeled, batch_size):
        batches = []

        index_shuf = np.arange(len(user_item_pairs))
        np.random.shuffle(index_shuf)
        _user_item_pairs = user_item_pairs[index_shuf]
        for i in range(int(len(_user_item_pairs) / batch_size)):

            ui_pairs = _user_item_pairs[i * batch_size: (i + 1) * batch_size, :]
            ui_pairs = ui_pairs.astype('int32')

            label = np.asarray(rating_matrix[ui_pairs[:, 0], ui_pairs[:, 1]])[0]
            label[label == -1] = 0

            unlabeled_idx = np.random.choice(np.arange(num_unlabeled), size=np.int(batch_size))

            train_batch = np.r_[ui_pairs, unlabeled_ui[unlabeled_idx, :2]]
            train_label = np.r_[label, unlabeled_ui[unlabeled_idx, 2]]

            batches.append([train_batch[:, 0], train_batch[:, 1], train_label])

        return batches

    @staticmethod
    def get_unobs_batches(user_item_pairs, rating_matrix, unlabeled_ui, num_unlabeled, batch_size):
        batches = []

        index_shuf = np.arange(len(user_item_pairs))
        np.random.shuffle(index_shuf)
        _user_item_pairs = user_item_pairs[index_shuf]
        for i in range(int(len(_user_item_pairs) / batch_size)):
            ui_pairs = _user_item_pairs[i * batch_size: (i + 1) * batch_size, :]
            ui_pairs = ui_pairs.astype('int32')

            label = np.asarray(rating_matrix[ui_pairs[:, 0], ui_pairs[:, 1]])[0]
            label[label == -1] = 0

            unlabeled_idx = np.random.choice(np.arange(num_unlabeled), size=np.int(2 * batch_size))

            # train_batch = np.r_[ui_pairs, unlabeled_ui[unlabeled_idx, :2]]
            # train_label = np.r_[label, unlabeled_ui[unlabeled_idx, 2]]
            train_batch = unlabeled_ui[unlabeled_idx, :2]
            train_label = unlabeled_ui[unlabeled_idx, 2]

            batches.append([train_batch[:, 0], train_batch[:, 1], train_label])

        return batches

    @staticmethod
    def get_all_batches(num_users, num_items, batch=True):
        r0 = np.arange(num_users)
        r1 = np.arange(num_items)

        out = np.empty((num_users, num_items, 2), dtype=np.dtype(np.int32))
        out[:, :, 0] = r0[:, None]
        out[:, :, 1] = r1

        all_pairs = out.reshape(-1, 2)

        if batch:
            batches = []

            for i in range(int(len(all_pairs) / num_items)):
                ui_pairs = all_pairs[i * num_items: (i + 1) * num_items, :]
                ui_pairs = ui_pairs.astype('int32')

                batches.append([ui_pairs[:, 0], ui_pairs[:, 1]])

            return batches
        else:
            return all_pairs

    def train_model(self, matrix_train, matrix_valid, matrix_test, epoch=100, metric='AUC', topK=50,
                    is_topK=False, gpu_on=False, seed=0):
        np.random.seed(seed)
        tf.set_random_seed(seed)

        user_item_matrix = lil_matrix(matrix_train)
        user_item_pairs = np.asarray(user_item_matrix.nonzero(), order='F').T

        valid_user_item_matrix = lil_matrix(matrix_valid)
        valid_user_item_pairs = np.asarray(valid_user_item_matrix.nonzero()).T.astype('int32')

        test_user_item_matrix = lil_matrix(matrix_test)
        test_user_item_pairs = np.asarray(test_user_item_matrix.nonzero()).T.astype('int32')

        all_batches = self.get_all_batches(self._num_users, self._num_items)

        temp_matrix_train = csr_matrix(matrix_train.shape)
        temp_matrix_train[matrix_train.nonzero()] = 1

        all_ui_pair = self.get_all_batches(self._num_users, self._num_items, False)
        user_item_pairs_rows = user_item_pairs.view([('', user_item_pairs.dtype)] * user_item_pairs.shape[1])
        all_ui_pair_rows = all_ui_pair.view([('', all_ui_pair.dtype)] * all_ui_pair.shape[1])
        unlabeled_ui_pair = np.setdiff1d(
            all_ui_pair_rows, user_item_pairs_rows).view(all_ui_pair.dtype).reshape(-1, all_ui_pair.shape[1])

        train_ui = np.r_[np.c_[user_item_pairs, np.ones(user_item_pairs.shape[0])],
                         np.c_[unlabeled_ui_pair, np.zeros(unlabeled_ui_pair.shape[0])]].astype('int32')

        unlabeled_train = train_ui[train_ui[:, 2] == 0]
        num_unlabeled = np.sum(1 - train_ui[:, 2])

        # Training
        best_result, best_topk_prediction, best_vrating_prediction, best_trating_prediction = 0, None, None, None
        result_early_stop = 0
        for i in tqdm(range(epoch)):
            a, b, c = [], [], []
            batches = self.get_batches(user_item_pairs, matrix_train, unlabeled_train, num_unlabeled, self._batch_size)
            for step in range(len(batches)):
                feed_dict = {self.user_idx: batches[step][0],
                             self.item_idx: batches[step][1],
                             self.label: batches[step][2]
                             }
                _, ta, tb, tc = self.sess.run([self._train, self.a, self.b, self.c], feed_dict=feed_dict)
                a.append(ta)
                b.append(tb)
                c.append(tc)

            # unobs_batches = self.get_unobs_batches(user_item_pairs, matrix_train, unlabeled_train,
            #                                        num_unlabeled, self._batch_size)
            # for step in range(len(batches)):
            #     feed_dict = {self.user_idx: unobs_batches[step][0],
            #                  self.item_idx: unobs_batches[step][1],
            #                  self.label: unobs_batches[step][2]
            #                  }
            #     _ = self.sess.run([self._train], feed_dict=feed_dict)

            valid_rating_prediction = self.sess.run(self.sigmoid_z_x_ij,
                                                    feed_dict={self.user_idx: valid_user_item_pairs[:, 0],
                                                               self.item_idx: valid_user_item_pairs[:, 1]})

            all_prediction = []
            for step in range(len(all_batches)):
                feed_dict = {self.user_idx: all_batches[step][0],
                             self.item_idx: all_batches[step][1],
                             }
                vector_predict = self.sess.run([self.sigmoid_z_x_ij], feed_dict=feed_dict)

                all_prediction.append(vector_predict)

            all_prediction = np.vstack(all_prediction) + -99999. * temp_matrix_train.toarray()
            _, topk_prediction = self.sess.run(self.top_k, feed_dict={self.prediction_top_k: all_prediction,
                                                                      self.scale_top_k: topK})

            valid_result = evaluate(valid_rating_prediction, topk_prediction, matrix_valid, [metric], [topK],
                                    is_topK=is_topK)

            test_rating_prediction = self.sess.run(self.sigmoid_z_x_ij,
                                                   feed_dict={self.user_idx: test_user_item_pairs[:, 0],
                                                              self.item_idx: test_user_item_pairs[:, 1]})
            test_result = evaluate(test_rating_prediction, topk_prediction, matrix_test, [metric], [topK],
                                   is_topK=is_topK)

            if valid_result[metric][0] > best_result:
                best_result = valid_result[metric][0]
                best_topk_prediction = topk_prediction
                best_vrating_prediction, best_trating_prediction = valid_rating_prediction, test_rating_prediction
                result_early_stop = 0
            else:
                result_early_stop += 1
                if result_early_stop > 5:
                    break

            for _metric in valid_result.keys():
                print("Epoch {0} Valid {1}:{2}".format(i, _metric, valid_result[_metric]))

            for _metric in test_result.keys():
                print("Epoch {0} Test {1}:{2}".format(i, _metric, test_result[_metric]))

            print("Epoch {0} a {1} b {2} c {3}".format(i, sum(a) / len(a), sum(b) / len(b), sum(c) / len(c)))

        return best_result, best_topk_prediction, best_vrating_prediction, best_trating_prediction


def dibncf(matrix_train, matrix_valid, matrix_test, embeded_matrix=np.empty(0), iteration=1000, lam=0.01, alpha=0.01,
           rank=200, gamma=0.01, batch_size=500, learning_rate=1e-3, optimizer="Adam", seed=0, gpu_on=False,
           metric='AUC', topK=50, is_topK=False, searcher='optuna', n_trials=100, **unused):
    progress = WorkSplitter()

    progress.section("DIB-NCF: Set the random seed")
    np.random.seed(seed)
    tf.set_random_seed(seed)

    progress.section("DIB-NCF: Training")
    matrix_input = matrix_train
    if embeded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embeded_matrix.T))

    m, n = matrix_input.shape

    if searcher == 'optuna':
        tuner = Tuner()
        trials, best_params = tuner.tune(n_trials=n_trials, num_users=m, num_items=n, optimizer=Regularizer[optimizer],
                                         gpu_on=gpu_on, train=matrix_input, valid=matrix_valid, test=matrix_test,
                                         epoch=iteration, metric=metric, topK=topK, is_topK=is_topK, seed=seed)
        return trials, best_params

    if searcher == 'grid':
        model = DIBNCF(m, n, rank, batch_size, lamb=lam, alpha=alpha, gamma=gamma, learning_rate=learning_rate,
                       optimizer=Regularizer[optimizer], gpu_on=gpu_on)

        result, topk_prediction, valid_rating_prediction, test_rating_prediction = model.train_model(matrix_input,
                                                                                                     matrix_valid,
                                                                                                     matrix_test,
                                                                                                     iteration, metric,
                                                                                                     topK, is_topK,
                                                                                                     gpu_on, seed)

        model.sess.close()
        tf.reset_default_graph()

        return result, topk_prediction, valid_rating_prediction, test_rating_prediction