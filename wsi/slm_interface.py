from typing import List, Tuple, Dict
from .WSISettings import WSISettings

class SLM:
    """
    implement this interface for to use a custom biLM
    """

    def __init__(self):
        pass

    def predict_sent_substitute_representatives(self, inst_id_to_sentence: Dict[str, Tuple[List[str], int]],wsisettings:WSISettings) \
            -> Dict[str, List[Dict[str, int]]]:
        raise NotImplementedError()
