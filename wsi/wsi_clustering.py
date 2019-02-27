from typing import Dict, List, Tuple
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from collections import Counter
import numpy as np
import logging
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize

import pickle

# with open('gold_n_senses.pickle', 'rb') as fin:
#     gold_n_senses = pickle.load(fin)


def cluster_inst_ids_representatives(inst_ids_to_representatives: Dict[str, List[Dict[str, int]]],
                                     max_number_senses: float,min_sense_instances:int,
                                     disable_tfidf: bool, explain_features: bool) -> Tuple[
    Dict[str, Dict[str, int]], List]:
    global gold_n_senses
    """
    preforms agglomerative clustering on representatives of one SemEval target
    :param inst_ids_to_representatives: map from SemEval instance id to list of representatives
    :param n_clusters: fixed number of clusters to use
    :param disable_tfidf: disable tfidf processing of feature words
    :return: map from SemEval instance id to soft membership of clusters and their weight
    """
    inst_ids_ordered = list(inst_ids_to_representatives.keys())
    lemma = inst_ids_ordered[0].rsplit('.', 1)[0]
    logging.info('clustering lemma %s' % lemma)
    representatives = [y for x in inst_ids_ordered for y in inst_ids_to_representatives[x]]
    n_represent = len(representatives) // len(inst_ids_ordered)
    dict_vectorizer = DictVectorizer(sparse=False)
    rep_mat = dict_vectorizer.fit_transform(representatives)
    # to_pipeline = [dict_vectorizer]
    if disable_tfidf:
        transformed = rep_mat
    else:
        transformed = TfidfTransformer(norm=None).fit_transform(rep_mat).todense()


    metric = 'cosine'
    method = 'average'
    dists = pdist(transformed, metric=metric)
    Z = linkage(dists, method=method, metric=metric)

    distance_crit = Z[-max_number_senses, 2]

    labels = fcluster(Z, distance_crit,
                      'distance') - 1

    n_senses = np.max(labels) + 1

    senses_n_domminates = Counter()
    instance_senses = {}
    for i, inst_id in enumerate(inst_ids_ordered):
        inst_id_clusters = Counter(labels[i * n_represent:
                                          (i + 1) * n_represent])
        instance_senses[inst_id] = inst_id_clusters
        senses_n_domminates[inst_id_clusters.most_common()[0][0]] += 1

    big_senses = [x for x in senses_n_domminates if senses_n_domminates[x] >= min_sense_instances]

    sense_means = np.zeros((n_senses, transformed.shape[1]))
    for sense_idx in range(n_senses):
        idxs_this_sense = np.where(labels == sense_idx)
        cluster_center = np.mean(np.array(transformed)[idxs_this_sense], 0)
        sense_means[sense_idx] = cluster_center

    sense_remapping = {}
    if min_sense_instances > 0:
        dists = cdist(sense_means, sense_means, metric='cosine')
        closest_senses = np.argsort(dists, )[:, ]

        for sense_idx in range(n_senses):
            for closest_sense in closest_senses[sense_idx]:
                if closest_sense in big_senses:
                    sense_remapping[sense_idx] = closest_sense
                    break
        new_order_of_senses = list(set(sense_remapping.values()))
        sense_remapping = dict((k, new_order_of_senses.index(v)) for k, v in sense_remapping.items())

        labels = np.array([sense_remapping[x] for x in labels])


    best_instance_for_sense = {}
    senses = {}
    for inst_id, inst_id_clusters in instance_senses.items():
        senses_inst = {}
        for sense_idx, count in inst_id_clusters.most_common():
            if sense_remapping:
                sense_idx = sense_remapping[sense_idx]
            senses_inst[f'{lemma}.sense.{sense_idx}'] = count
            if sense_idx not in best_instance_for_sense:
                best_instance_for_sense[sense_idx] = (count, inst_id)
            else:
                current_count, current_best_inst = best_instance_for_sense[sense_idx]
                if current_count < count:
                    best_instance_for_sense[sense_idx] = (count, inst_id)

        senses[inst_id] = senses_inst

    label_count = Counter(labels)
    statistics = []
    if len(label_count) > 1 and explain_features:
        svm = LinearSVC(class_weight='balanced', penalty='l1', dual=False)
        svm.fit(rep_mat, labels)

        coefs = svm.coef_
        top_coefs = np.argpartition(coefs, -10)[:, -10:]
        if top_coefs.shape[0] == 1:
            top_coefs = [top_coefs[0], -top_coefs[0]]

        rep_arr = np.asarray(rep_mat)
        totals_cols = rep_arr.sum(0)

        p_feats = totals_cols / transformed.shape[0]

        for sense_idx, top_coef_sense in enumerate(top_coefs):
            count_reps = label_count[sense_idx]
            # p_sense = count_reps / transformed.shape[0]
            count_sense_feat = rep_arr[np.where(labels == sense_idx)]
            p_sense_feat = count_sense_feat.sum(0) / transformed.shape[0]

            pmis_proxy = p_sense_feat / (p_feats + 0.00000001)
            best_features_pmi_idx = np.argpartition(pmis_proxy, -10)[-10:]

            closest_instance = best_instance_for_sense[sense_idx][1]
            best_features = [dict_vectorizer.feature_names_[x] for x in top_coef_sense]
            best_features_pmi = [dict_vectorizer.feature_names_[x] for x in best_features_pmi_idx]

            # logging.info(f'sense #{sense_idx+1} ({label_count[sense_idx]} reps) best features: {best_features}')
            statistics.append((count_reps, best_features, best_features_pmi, closest_instance))

    return senses, statistics
