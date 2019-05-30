### Towards better substitution-based word sense induction - Word Sense Induction with BERT


a follow up to https://github.com/asafamr/SymPatternWSI , adapted to BERT.<br>


paper: Towards better substitution-based word sense induction - https://arxiv.org/abs/1905.12598

### prerequisites:
Python 3.7<br>
install requirements.txt with pip -r<br>
This will install python pacakges including pytorch and huggingface's BERT port.<br>
(for CUDA support first install pytorch accroding to [their instructions](https://pytorch.org/))<br>


run download_resources.sh to download datasets.


### WSI:
run wsi_bert.py for sense induction on both SemEval 2010 and 2013 WSI task datasets. <br>
Logs should be printed to "debug" dir. 

### results - (SOTA when published):

SemEval 2013 WSI mean(STD) over 10 runs:<br>
FNMI:21.4(0.5)  FBC:64.0(0.5)  Geom. mean:37.0(0.5)<br>
(previous SOTA 11.3,57.5,25.4)

<br>

SemEval 2010 WSI mean(STD) over 10 runs:<br>
F-S:71.3(0.1) V-M:40.4(1.8)  Geom. mean:53.6(1.2)<br>
(previous SOTA 61.7,9.8,24.59)
