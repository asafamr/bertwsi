from wsi.lm_bert import LMBert
import os
import logging
from time import strftime
from wsi.WSISettings import DEFAULT_PARAMS, WSISettings
from wsi.wsi import WordSenseInductor
from multiprocessing import cpu_count
# from pytorch_pretrained_bert import *
import sys

if __name__ == '__main__':

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(logging.NullHandler())
    # for i in range(10):
    settings = DEFAULT_PARAMS._asdict()

    # --------------- modify default settings

    # settings['debug_dir'] = 'draft'
    #
    # run_name = strftime("%m%d-%H%M%S") + '-wth72'
    # settings['run_name'] = run_name

    # --------------- finalizing settings
    settings = WSISettings(**settings)

    startmsg = 'BERT WSI Demo\n\n'
    startmsg += 'Arguments:\n'
    startmsg += '-' * 10 + '\n'
    for arg, val in settings._asdict().items():
        startmsg += (' %-30s:%s\n' % (arg.replace('_', '-'), val))
    startmsg = startmsg.strip()

    lm = LMBert(settings.cuda_device, settings.bert_model,
                max_batch_size=settings.max_batch_size)

    if settings.debug_dir:
        if not os.path.exists(settings.debug_dir):
            os.makedirs(settings.debug_dir)

        # root_logger.disabled=True
        handler = logging.FileHandler(os.path.join(settings.debug_dir, '%s.log.txt' % settings.run_name), 'w', 'utf-8')
        formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

        # handler2 = logging.StreamHandler()
        # handler2.setFormatter(formatter)
        # root_logger.addHandler(handler2)
    logging.info(startmsg)

    if sys.platform == 'linux':
        os.popen(f"taskset -cp 0-{cpu_count()-1} {os.getpid()}").read()  # scipy defaults to one core otherwise

    print(startmsg)
    print('this run name: %s' % settings.run_name)

    word_sense_inductor = WordSenseInductor(lm)

    scores2010, scores2013 = word_sense_inductor.run(settings,
                                                     print_progress=True)
    logging.info('full results: %s' % ((scores2013,scores2010),))
