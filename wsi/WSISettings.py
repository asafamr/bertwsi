from collections import namedtuple

WSISettings = namedtuple('WSISettings', ['n_represents', 'n_samples_per_rep', 'cuda_device', 'debug_dir',
                                         'disable_tfidf', 'disable_lemmatization', 'run_name','patterns',
                                         'min_sense_instances','bert_model',
                                         'max_batch_size', 'prediction_cutoff', 'max_number_senses',
                                         ])

DEFAULT_PARAMS = WSISettings(
    n_represents=15,
    n_samples_per_rep=20,
    cuda_device=2,
    debug_dir='debug',
    disable_lemmatization=False,
    disable_tfidf=False,
    patterns=[('{pre} {target_predict} {post}', 0.5)],
    run_name='',
    max_number_senses=10,
    min_sense_instances=2,
    max_batch_size=10,
    prediction_cutoff=200,
    bert_model='bert-large-cased'
)
