from transformers import AdamW
import torch
from model.modeling_llama2 import LlamaForCausalLM
from model.configuration_llama2 import LlamaConfig
from tokenizer.tokenization_llama import LlamaTokenizer
from argparse import ArgumentParser
from typing import Dict
from datasets import load_dataset
from utils import create_attn_mask, mask_pads
from moreh.driver.common import config

def parse_args():
    parser = ArgumentParser(description="LLaMA2 FineTuning")
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        help="model name or path",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="train bacth size")
    parser.add_argument("--block-size", type=int, default=1024, help="max input token length")
    parser.add_argument("--lr", type=float, default=0.00001, help="learning rate")
    

    parser.add_argument("--log-interval", type=int, default=10, help="log interval")
    parser.add_argument(
        "--save-model-dir", type=str, default="./", help="path to save model"
    )
    args = parser.parse_args()

    return args


def main(args):

    torch.moreh.option.enable_advanced_parallelization()

    model_args = LlamaConfig.from_pretrained(args.model_name_or_path)
    model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, config=model_args)

    model.cuda()
    model.train()

    tokenizer = LlamaTokenizer.from_pretrained(
        args.model_name_or_path,
    )

    tokenizer.pad_token_id = 0

    def preprocess(example):
        input_ids = tokenizer(
            example["text"],
            return_attention_mask=False,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            max_length=args.block_size,
        )["input_ids"]
        return {"input_ids": input_ids}

    train_dataset = load_dataset("tatsu-lab/alpaca").with_format("torch")
    train_dataset = train_dataset.map(preprocess)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    optim = AdamW(model.parameters(), lr=args.lr)

    total_step = len(train_dataloader)

    for i, batch in enumerate(train_dataloader):
        batch = batch["input_ids"]
        inputs, labels = batch, mask_pads(batch, tokenizer)
        attn_mask = create_attn_mask(inputs, tokenizer)

        outputs = model(
            inputs.cuda(),
            labels=labels.cuda(),
            use_cache=False,
            attention_mask=attn_mask.cuda(),
        )
        loss = outputs[0]
        loss.backward()

        optim.step()
        model.zero_grad(set_to_none=True)
            
        if i % args.log_interval == 0:
            print(f"[Step {i}/{total_step}] Loss: {loss.item()}")
            
    model.save_pretrained(args.save_model_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
