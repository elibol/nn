import numpy as np


def singlelabel_error(Y_truth, Y_pred):
    return np.mean(np.argmax(Y_truth, axis=1) != np.argmax(Y_pred, axis=1))


def multilabel_performance_labels(readable=True):
    if readable:
        return [
            'true_pos/num_guessed',
            'false_pos/num_guessed',
            'true_pos/num_truth',
            'false_neg/num_truth',
        ]
    return [
        'tp/(tp+fp)',
        'fp/(tp+fp)',
        'tp/(tp+fn)',
        'fn/(tp+fn)',
    ]


def multilabel_performance(Y_truth, Y_pred, features_obj=None, mean=True):
    result = []

    # import ipdb; ipdb.set_trace()
    num_samples = Y_truth.shape[0]
    for i in range(num_samples):

        if type(Y_pred) is np.ndarray:
            truth_vec = Y_truth[i]
        else:
            truth_vec = Y_truth[i].toarray()[0]

        if type(Y_pred) is np.ndarray:
            guess_vec = Y_pred[i]
        else:
            guess_vec = Y_pred[i].toarray()[0]

        if truth_vec.dtype is not np.bool:
            truth_vec = truth_vec > 0

        # scoring hack
        # import ipdb; ipdb.set_trace()
        guess_vec_args = np.argsort(guess_vec)[::-1]
        guess_vec_nonzero = guess_vec_args[:np.sum(truth_vec)]
        guess_vec_binary = np.zeros(guess_vec.shape[0], dtype=np.bool)
        guess_vec_binary[guess_vec_nonzero] = True

        if features_obj:
            truth_set = features_obj.label_vector_to_cui_set(truth_vec)
            guess_set = features_obj.label_vector_to_cui_set(guess_vec_binary)
            # print truth_set, guess_set, clinician_set
        else:
            truth_set = frozenset(np.where(truth_vec)[0])
            guess_set = frozenset(np.where(guess_vec_binary)[0])

        num_truth = float(len(truth_set))
        num_guessed = float(len(guess_set))
        tp = truth_set.intersection(guess_set)
        fp = guess_set - truth_set
        fn = truth_set - guess_set
        # num_truth = tp + fn
        # num_guessed = tp + fp

        assert num_guessed == len(tp) + len(fp)
        assert num_truth == len(tp) + len(fn)

        if num_guessed == 0:
            tpg_frac = 0
            fpg_frac = 0
        else:
            tpg_frac = len(tp) / num_guessed
            fpg_frac = len(fp) / num_guessed

        if num_truth == 0:
            tpt_frac = 0
            fnt_frac = 0
        else:
            tpt_frac = len(tp) / num_truth
            fnt_frac = len(fn) / num_truth

        result.append([tpg_frac, fpg_frac,
                       tpt_frac, fnt_frac])

    if mean:
        return np.mean(result, axis=0)
    return result
