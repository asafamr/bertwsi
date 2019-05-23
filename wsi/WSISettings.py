from collections import namedtuple

WSISettings = namedtuple('WSISettings', ['n_represents', 'n_samples_per_rep', 'cuda_device', 'debug_dir',
                                         'disable_tfidf', 'disable_lemmatization', 'run_name', 'patterns',
                                         'min_sense_instances', 'bert_model',
                                         'max_batch_size', 'prediction_cutoff', 'max_number_senses',
                                         ])

DEFAULT_PARAMS = WSISettings(
    n_represents=15,
    n_samples_per_rep=20,
    cuda_device=1,
    debug_dir='debug',
    disable_lemmatization=False,
    disable_tfidf=False,
    patterns=[('{pre} {target} (or even {mask_predict}) {post}', 0.4),
              ('{pre} {target_predict} {post}', 0.4)],
    # (pattern,weight): each of these patterns will produce a prediction state.
    # the weighted sum of them will be matmul'ed for a distribution over substitutes

    # patterns=[('{pre} {target_predict} {post}', 0.5)], # - just predict on first token, no patterns

    run_name='test-run',
    max_number_senses=7,
    min_sense_instances=2,
    # sense clusters that dominate less than this number of samples
    #  would be remapped to their closest big sense

    max_batch_size=10,
    prediction_cutoff=200,
    bert_model='bert-large-uncased'
)
