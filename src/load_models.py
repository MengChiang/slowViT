from slowfast.config.defaults import get_cfg
from slowfast.models import build_model
from slowfast.utils.checkpoint import load_checkpoint
from transformers import BertModel, BertTokenizer

def load_models():
    # Load pre-trained SlowFast model
    cfg = get_cfg()
    cfg.merge_from_file('configs/Kinetics/C2D_8x8_R50.yaml')
    cfg.MODEL.WEIGHTS = 'checkpoints/checkpoint_epoch_200.pyth'
    slowfast_model = build_model(cfg)
    load_checkpoint(cfg.MODEL.WEIGHTS, slowfast_model, cfg.NUM_GPUS > 1, None, inflation=False)
    slowfast_model.eval()

    # Load pre-trained BERT model
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    return slowfast_model, bert_model