from .slm_interface import SLM
from wsi.semeval_utils import generate_sem_eval_2013_no_tokenization, generate_sem_eval_2010_no_tokenization, \
    evaluate_labeling_2010, evaluate_labeling_2013,get_n_senses_corr
from collections import defaultdict
from .wsi_clustering import cluster_inst_ids_representatives
from tqdm import tqdm
import logging
from .WSISettings import WSISettings
import os
import numpy as np


class WordSenseInductor:
    def __init__(self, lm: SLM):
        self.bilm = lm

    def _perform_wsi_on_ds_gen(self, ds_name, gen, wsisettings: WSISettings, eval_proc, print_progress=False):
        ds_by_target = defaultdict(dict)
        for pre, target, post, inst_id in gen:
            lemma_pos = inst_id.rsplit('.', 1)[0]
            ds_by_target[lemma_pos][inst_id] = (pre, target, post)

        inst_id_to_sense = {}
        gen = ds_by_target.items()
        if print_progress:
            gen = tqdm(gen, desc=f'predicting substitutes {ds_name}')
        for lemma_pos, inst_id_to_sentence in gen:
            inst_ids_to_representatives = \
                self.bilm.predict_sent_substitute_representatives(inst_id_to_sentence=inst_id_to_sentence,
                                                                  wsisettings=wsisettings)

            clusters, statistics = cluster_inst_ids_representatives(
                inst_ids_to_representatives=inst_ids_to_representatives,
                max_number_senses=wsisettings.max_number_senses,min_sense_instances=wsisettings.min_sense_instances,
                disable_tfidf=wsisettings.disable_tfidf,explain_features=True)
            inst_id_to_sense.update(clusters)
            if statistics:
                logging.info('Sense cluster statistics:')
                for idx, (rep_count, best_features,best_features_pmi, best_instance_id) in enumerate(statistics):
                    best_instance = ds_by_target[lemma_pos][best_instance_id]
                    nice_print_instance = f'{best_instance[0]} -{best_instance[1]}- {best_instance[2]}'
                    logging.info(
                        f'Sense {idx}, # reps: {rep_count}, best feature words: {", ".join(best_features)}.'
                        f', best feature words(PMI): {", ".join(best_features_pmi)}.'
                        f' closest instance({best_instance_id}):\n---\n{nice_print_instance}\n---\n')

        out_key_path = None
        if wsisettings.debug_dir:
            out_key_path = os.path.join(wsisettings.debug_dir, f'{wsisettings.run_name}-{ds_name}.key')

        if print_progress:
            print(f'writing {ds_name} key file to %s' % out_key_path)

        return eval_proc(inst_id_to_sense, out_key_path)


    @staticmethod
    def _get_score_by_pos(results):
        res_string = ''
        for pos, pos_title in [('v', 'VERB'), ('n', 'NOUN'), ('j', 'ADJ')]:
            aggregated = defaultdict(list)
            for lemmapos in results:
                if lemmapos[-2:] == '.' + pos:
                    for metric, score in results[lemmapos].items():
                        aggregated[metric].append(score)
            if aggregated:
                avg = 1
                for metric, listscores in aggregated.items():
                    avg *= np.mean(listscores)
                avg = np.sqrt(avg)
                res_string += (f'{pos_title} ' + ' '.join(
                    [f'{metric}: {np.mean(listscores)*100:.2f}' for metric, listscores in aggregated.items()]))
                res_string += f' AVG: {avg*100:.2f}\n'
        return res_string


    def run(self, wsisettings: WSISettings,
            print_progress=False):

        # SemEval target might be, for example, book.n (lemma+POS)
        # SemEval instance might be, for example, book.n.12 (target+index).
        # In the example instance above, corresponds to one usage of book as a noun in a sentence

        # semeval_dataset_by_target is a dict from target to dicts of instances with their sentence
        # so semeval_dataset_by_target['book.n']['book.n.12'] is the sentence tokens of the 'book.n.12' instance
        # and the index of book in these tokens

        scores2013,corr = self._perform_wsi_on_ds_gen(
            'SemEval2013',
            generate_sem_eval_2013_no_tokenization('./resources/SemEval-2013-Task-13-test-data'),
            wsisettings,
            lambda inst2sense, outkey:
            evaluate_labeling_2013('./resources/SemEval-2013-Task-13-test-data', inst2sense, outkey),
            print_progress=print_progress)

        fnmi = scores2013['all']['FNMI']
        fbc = scores2013['all']['FBC']
        msg = 'SemEval 2013 FNMI %.2f FBC %.2f AVG %.2f' % (fnmi * 100, fbc * 100, np.sqrt(fnmi * fbc) * 100)
        msg+='\n' + WordSenseInductor._get_score_by_pos(scores2013)
        for pos in corr:
            msg+= f'# senses correlation(p-value) {pos}: {corr[pos][0]*100:.2f}({corr[pos][1]*100:.2f})\n'
        logging.info(msg)
        if print_progress:
            print(msg)

        scores2010,corr = self._perform_wsi_on_ds_gen(
            'SemEval2010',
            generate_sem_eval_2010_no_tokenization('./resources/SemEval-2010/test_data'),
            wsisettings,
            lambda inst2sense, outkey:
            evaluate_labeling_2010('./resources/SemEval-2010', inst2sense, outkey),
            print_progress=print_progress)

        fscore = scores2010['all']['FScore']
        v_measure = scores2010['all']['V-Measure']

        msg = 'SemEval 2010 FScore %.2f V-Measure %.2f AVG %.2f' % (
            fscore * 100, v_measure * 100, np.sqrt(fscore * v_measure) * 100)
        msg += '\n' + WordSenseInductor._get_score_by_pos(scores2010)
        for pos in corr:
            msg+= f'# senses correlation(p-value) {pos}: {corr[pos][0]*100:.2f}({corr[pos][1]*100:.2f})\n'
        logging.info(msg)
        if print_progress:
            print(msg)

        return scores2010['all'], scores2013['all']
