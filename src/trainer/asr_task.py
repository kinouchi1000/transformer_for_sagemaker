import argparse
import logging
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch


from model.abs_asr_model import AbsModel
from model.asr_model import ASRModel
from model.encoder.abs_encoder import AbsEncoder
from model.encoder.transformer_encoder import TransformerEncoder
from model.decoder.abs_decoder import AbsDecoder
from model.decoder.transformer_decoder import TransformerDecoder
from model.frontend.abs_frontend import AbsFrontend
from model.frontend.default import DefaultFrontend
from model.preencoder.abs_preencoder import AbsPreEncoder
from model.postencoder.abs_postencoder import AbsPostEncoder
from model.ctc import CTC

from model.specaug.abs_specaug import AbsSpecAug
from model.specaug.specaug import SpecAug

from model.layers.abs_normalize import AbsNormalize

# test
specaug = AbsSpecAug()
normalize = AbsNormalize()
preencoder = AbsPreEncoder()
postencoder = AbsPostEncoder()

encoder = AbsEncoder()
decoder = AbsDecoder()
ctc = CTC(1, 1)

model = ASRModel(
    vocab_size=2,
    token_list=["a", "b"],
    specaug=specaug,
    normalize=normalize,
    preencoder=preencoder,
    encoder=encoder,
    decoder=decoder,
    ctc=ctc,
)

