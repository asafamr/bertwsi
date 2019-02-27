import spacy
import os
from xml.etree import ElementTree
from typing import Dict,Tuple
import tempfile
import subprocess
import logging
import pickle
from collections import defaultdict
from scipy.stats import spearmanr


def generate_sem_eval_2013(dir_path: str):
    logging.info('reading SemEval dataset from %s' % dir_path)
    nlp = spacy.load("en", disable=['ner', 'parser'])
    in_xml_path = os.path.join(dir_path, 'contexts/senseval2-format/semeval-2013-task-13-test-data.senseval2.xml')
    gold_key_path = os.path.join(dir_path, 'keys/gold/all.key')
    with open(in_xml_path, encoding="utf-8") as fin_xml, open(gold_key_path, encoding="utf-8") as fin_key:
        instid_in_key = set()
        for line in fin_key:
            lemma_pos, inst_id, _ = line.strip().split(maxsplit=2)
            instid_in_key.add(inst_id)
        et_xml = ElementTree.parse(fin_xml)
        for word in et_xml.getroot():
            for inst in word.getchildren():
                inst_id = inst.attrib['id']
                if inst_id not in instid_in_key:
                    # discard unlabeled instances
                    continue
                context = inst.find("context")
                before, target, after = list(context.itertext())
                before = [x.text for x in nlp(before.strip(), disable=['parser', 'tagger', 'ner'])]
                target = target.strip()
                after = [x.text for x in nlp(after.strip(), disable=['parser', 'tagger', 'ner'])]
                yield before + [target] + after, len(before), inst_id


def generate_sem_eval_2013_no_tokenization(dir_path: str):
    logging.info('reading SemEval dataset from %s' % dir_path)
    in_xml_path = os.path.join(dir_path, 'contexts/senseval2-format/semeval-2013-task-13-test-data.senseval2.xml')
    gold_key_path = os.path.join(dir_path, 'keys/gold/all.key')
    with open(in_xml_path, encoding="utf-8") as fin_xml, open(gold_key_path, encoding="utf-8") as fin_key:
        instid_in_key = set()
        for line in fin_key:
            lemma_pos, inst_id, _ = line.strip().split(maxsplit=2)
            instid_in_key.add(inst_id)
        et_xml = ElementTree.parse(fin_xml)
        for word in et_xml.getroot():
            for inst in word.getchildren():
                inst_id = inst.attrib['id']
                if inst_id not in instid_in_key:
                    # discard unlabeled instances
                    continue
                context = inst.find("context")
                before, target, after = list(context.itertext())
                # before = [x.text for x in nlp(before.strip(), disable=['parser', 'tagger', 'ner'])]
                # target = target.strip()
                # after = [x.text for x in nlp(after.strip(), disable=['parser', 'tagger', 'ner'])]
                yield before.strip(), target.strip(), after.strip(), inst_id


def generate_sem_eval_2010_no_tokenization(dir_path: str):
    logging.info('reading SemEval dataset from %s' % dir_path)
    cached = []
    cache_file_path = os.path.join(dir_path, 'wsi2010_cache.pickle')
    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'rb') as fin:
            cached = pickle.load(fin)
    else:
        nlp = spacy.load('en', disable=['ner'])
        additional_mapping = {'stuck': 'stick', 'straightened': 'straighten', 'shaving': 'shave', 'swam': 'swim',
                              'figger': 'figure',
                              'committed': 'commit', 'divided': 'divide', 'lie': 'lay', 'lay': 'lie', 'lah': 'lie',
                              'swimming': 'swim'}

        def basic_stem(w):
            if w[-1] == 's':
                w = w[:-1]
            elif w[-3:] == 'ing':
                w = w[:-3]
            elif w[-2:] == 'ed':
                w = w[:-2]
            return w

        for root_dir, dirs, files in os.walk(dir_path):  # "../paper-menuscript/resources/SemEval-2010/test_data/"):
            #     path = root.split(os.sep)
            for file in files:
                if '.xml' in file:
                    tree = ElementTree.parse(os.path.join(root_dir, file))
                    root = tree.getroot()
                    for child in root:
                        inst_name = child.tag
                        lemma = inst_name.split('.')[0]

                        stemmed_lemma = basic_stem(lemma)

                        # pres_sent = child.text
                        target_sent = child[0].text
                        post_sent = child[0].tail

                        # if not pres_sent:
                        #     pres_sent = ''
                        # if not post_sent:
                        #     post_sent = ''

                        parsed = nlp(target_sent)
                        first_occur_idx = None
                        for idx, w in enumerate(parsed):
                            token_lemma = basic_stem(
                                w.lemma_)  # we need to find the lemma withing our sentence - this does the trick
                            if token_lemma == stemmed_lemma or additional_mapping.get(w.lemma_) == lemma:
                                first_occur_idx = idx
                                break
                        if first_occur_idx is None:
                            print(file, [x.lemma_ for x in parsed], target_sent)

                        # pre = pres_sent + ' ' + ''.join(parsed[i].string for i in range(first_occur_idx))
                        pre =''.join(parsed[i].string for i in range(first_occur_idx))
                        ambig = parsed[first_occur_idx].text
                        post = ''.join(
                            parsed[i].string for i in range(first_occur_idx + 1, len(parsed)))# + ' ' + post_sent

                        pre = pre.replace(" 's ", "'s ")
                        post = post.replace(" 's ", "'s ")
                        cached.append((pre, ambig, post, inst_name))
        try:
            with open(cache_file_path, 'wb') as fout:
                pickle.dump(cached, fout)
        except Exception as e:
            logging.exception(e)
    return cached

def get_n_senses_corr(gold_key,new_key):
    senses_lemma_gold=defaultdict(set)
    senses_lemma_sys=defaultdict(set)
    for dic,key in [(senses_lemma_gold,gold_key),(senses_lemma_sys,new_key)]:
        with open(key,'r') as fin:
            for line in fin:
                split=line.split()
                lemma=split[0]
                senses = [x.split('/')[0] for x in split[2:]]
                dic[lemma].update(senses)
    oredered_lemmas=list(senses_lemma_gold.keys())
    corr = {}
    for ending,name in [('','all'),('v','VERB'),('n','NOUN'),('j','ADJ')]:
        g=[len(senses_lemma_gold[x]) for x in oredered_lemmas if x.endswith(ending)]
        n=[len(senses_lemma_sys[x]) for x in oredered_lemmas if x.endswith(ending)]
        c=spearmanr(g,n)
        corr[name]=c.correlation,c.pvalue
    return corr



def evaluate_labeling_2013(dir_path, labeling: Dict[str, Dict[str, int]], key_path: str = None) \
        -> Tuple[Dict[str, Dict[str, float]],Tuple]:
    """
    labeling example : {'become.v.3': {'become.sense.1':3,'become.sense.5':17} ... }
    means instance become.v.3' is 17/20 in sense 'become.sense.5' and 3/20 in sense 'become.sense.1'
    :param key_path: write produced key to this file
    :param dir_path: SemEval dir
    :param labeling: instance id labeling
    :return: FNMI, FBC as calculated by SemEval provided code
    """
    logging.info('starting evaluation key_path: %s' % key_path)

    def get_scores(gold_key, eval_key):
        ret = {}
        for metric, jar, column in [
            #         ('jaccard-index','SemEval-2013-Task-13-test-data/scoring/jaccard-index.jar'),
            #         ('pos-tau', 'SemEval-2013-Task-13-test-data/scoring/positional-tau.jar'),
            #         ('WNDC', 'SemEval-2013-Task-13-test-data/scoring/weighted-ndcg.jar'),
            ('FNMI', os.path.join(dir_path, 'scoring/fuzzy-nmi.jar'), 1),
            ('FBC', os.path.join(dir_path, 'scoring/fuzzy-bcubed.jar'), 3),
        ]:
            logging.info('calculating metric %s' % metric)
            res = subprocess.Popen(['java', '-jar', jar, gold_key, eval_key], stdout=subprocess.PIPE).stdout.readlines()
            # columns = []
            for line in res:
                line = line.decode().strip()
                if line.startswith('term'):
                    # columns = line.split('\t')
                    pass
                else:
                    split = line.split('\t')
                    if len(split) > column:
                        word = split[0]
                        # results = list(zip(columns[1:], map(float, split[1:])))
                        result = split[column]
                        if word not in ret:
                            ret[word] = {}
                        ret[word][metric] = float(result)

        return ret

    with tempfile.NamedTemporaryFile('wt') as fout:
        lines = []
        for instance_id, clusters_dict in labeling.items():
            clusters = sorted(clusters_dict.items(), key=lambda x: x[1])
            clusters_str = ' '.join([('%s/%d' % (cluster_name, count)) for cluster_name, count in clusters])
            lemma_pos = instance_id.rsplit('.', 1)[0]
            lines.append('%s %s %s' % (lemma_pos, instance_id, clusters_str))
        fout.write('\n'.join(lines))
        fout.flush()
        gold_key_path=os.path.join(dir_path, 'keys/gold/all.key')
        scores = get_scores(gold_key_path,
                            fout.name)
        if key_path:
            logging.info('writing key to file %s' % key_path)
            with open(key_path, 'w', encoding="utf-8") as fout2:
                fout2.write('\n'.join(lines))

        correlation = get_n_senses_corr(gold_key_path,fout.name)

    return scores,correlation


def evaluate_labeling_2010(dir_path, labeling: Dict[str, Dict[str, int]], key_path: str = None) \
        -> Tuple[Dict[str,Dict[str, float]],Tuple]:
    """
    similar to 2013 eval code, but only use top sense for each instnace
    """
    logging.info('starting evaluation key_path: %s' % key_path)
    unsup_key = os.path.join(dir_path, 'unsup_eval/keys/all.key')
    def get_scores(eval_key):
        ret = defaultdict(dict)


        for metric, jar in [
            ('FScore', os.path.join(dir_path, 'unsup_eval/fscore.jar')),
            ('V-Measure', os.path.join(dir_path, 'unsup_eval/vmeasure.jar'))
        ]:

            logging.info('calculating metric %s' % metric)
            res = subprocess.Popen(['java', '-jar', jar, eval_key, unsup_key, 'all'],
                                   stdout=subprocess.PIPE).stdout.readlines()
            for line in res:
                line = line.decode().strip()
                split = line.split()
                if len(split) == 4 and split[0][-2:] in ['.n', '.v', '.j']:

                    lemma = split[0]
                    score = float(split[1])
                    ret[lemma][metric] = score
                elif metric + ':' in line:
                    ret['all'][metric] = float(line.split(metric + ':')[1])
        return ret

    with tempfile.NamedTemporaryFile('wt') as fout:
        lines = []
        for instance_id, clusters_dict in labeling.items():
            clusters = sorted(clusters_dict.items(), key=lambda x: x[1])
            clusters_str = f'{clusters[-1][0]}'  # top sense
            lemma_pos = instance_id.rsplit('.', 1)[0]
            lines.append('%s %s %s' % (lemma_pos, instance_id, clusters_str))
        fout.write('\n'.join(lines))
        fout.flush()
        scores = get_scores(fout.name)
        if key_path:
            logging.info('writing key to file %s' % key_path)
            with open(key_path, 'w', encoding="utf-8") as fout2:
                fout2.write('\n'.join(lines))
            correlation = get_n_senses_corr(unsup_key, fout.name)
    return scores,correlation
