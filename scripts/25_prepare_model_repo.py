#!/usr/bin/env python3
"""Assemble the Hugging Face model directory from the local conversion output."""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--merged', type=Path, default=ROOT / 'build/merged')
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'hf-model')
    args = parser.parse_args()
    backbone = args.merged / 'kev-qwen3-0.6b-backbone'
    args.model_dir.mkdir(parents=True, exist_ok=True)
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
        shutil.copy2(backbone / name, args.model_dir / name)
    shutil.copy2(args.merged / 'kev-qwen3-0.6b-pointer-head.pt', args.model_dir / 'kev-qwen3-0.6b-pointer-head.pt')
    print(f'Model metadata copied to {args.model_dir}; Core ML package is created by script 21.')


if __name__ == '__main__':
    main()
