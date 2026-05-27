import argparse
import zipfile

import torch
from sconf import Config
from tqdm import tqdm

from datamodule import CROHMEDatamodule
from lit_comer import LitCoMER


def main(config_path: str, ckp_path: str, output_zip: str = "result.zip"):
    # Load config consistently with training
    config = Config(config_path)

    dm = CROHMEDatamodule(config=config)
    dm.setup("test")
    test_dataloader = dm.test_dataloader()

    # vocab_info is retrieved explicitly — no shared_vocab import side-effects
    vocab_info = dm.vocab.get_info()

    model = LitCoMER.load_from_checkpoint(ckp_path, config=config, vocab_info=vocab_info)
    model.eval()
    model.cuda()

    exprate_recorder = model.exprate_recorder

    with zipfile.ZipFile(output_zip, "w") as zip_f:
        with torch.inference_mode():
            for batch in tqdm(test_dataloader, desc="Testing"):
                batch = batch.to("cuda", non_blocking=True)

                # Inference — no grad, no training state
                hyps = model.approximate_joint_search(batch.imgs, batch.mask)
                exprate_recorder([h.seq for h in hyps], batch.indices)

                img_bases = batch.img_bases
                preds = [vocab_info.words.indices2label(h.seq) for h in hyps]

                # Write to zip incrementally
                for img_base, pred in zip(img_bases, preds):
                    content = f"%{img_base}\n${pred}$".encode()
                    with zip_f.open(f"{img_base}.txt", "w") as f:
                        f.write(content)

    exprate = exprate_recorder.compute()
    print(f"Validation ExpRate: {exprate}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--ckp", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="result.zip", help="Output zip file")
    args = parser.parse_args()

    main(args.config, args.ckp, args.output)
