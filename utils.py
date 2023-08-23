import copy

def create_attn_mask(input_ids, tokenizer):
    pad_token_ids = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return (input_ids != pad_token_ids).long()

def mask_pads(inputs, tokenizer, ignore_index = -100):
    idx_mask = create_attn_mask(inputs, tokenizer)
    labels = copy.deepcopy(inputs)
    labels[~idx_mask.bool()] = ignore_index
    return labels