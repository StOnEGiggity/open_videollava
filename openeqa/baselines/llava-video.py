# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import tqdm
from PIL import Image, PngImagePlugin

from openeqa.utils.prompt_utils import load_prompt
# from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default="data/hm3d-v0.json",
        help="path to EQA dataset (default: data/open-eqa-v0.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llava-video",
        help="Google model (default: gemini-pro-vision)",
    )
    parser.add_argument(
        "--frames-directory",
        type=Path,
        default="data/frames",
        help="path episode histories (default: data/frames)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=15,
        help="number of frames (default: 15)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="image size (default: 512)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default="data/results",
        help="output directory (default: data/results)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="continue running on API errors (default: false)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only process the first 5 questions",
    )
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.output_path = args.output_directory / (
        args.dataset.stem + "-{}.json".format(args.model)
    )
    return args


def parse_gemini_output(input: str, output: str) -> str:
    start_idx = output.find("A:")
    if start_idx == -1:
        return output.replace("A:", "").strip()
    end_idx = output.find("\n", start_idx)
    if end_idx == -1:
        return output[start_idx:].replace("A:", "").strip()
    return output[start_idx:end_idx].replace("A:", "").strip()


def ask_question(
    frame_paths: List,
    question: str,
    image_size: int,
    google_model: str,
    google_key: Optional[str] = None,
    force: bool = False,
    processor = None,
) -> Optional[str]:
    try:

        frames = [cv2.imread(p) for p in frame_paths]
        size = max(frames[0].shape)
        frames = [
            cv2.resize(img, dsize=None, fx=image_size / size, fy=image_size / size)
            for img in frames
        ]
        frames = np.stack([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in frames])

        prompt = load_prompt("llava-video")
        prefix, suffix = prompt.split("User Query:")
        # suffix = "User Query:" + suffix.format(question=question)

        # messages = []
        # messages += [prefix]
        # messages += frames
        # messages += [suffix]
        
        llava_prompt = "USER: <video>{question} ASSISTANT:".format(question=question)
        inputs = processor(text=llava_prompt, videos=frames, return_tensors="pt")

        # try:
        generate_ids = google_model.generate(**inputs, max_length=120)
        print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
        # except Exception as error:
        #     output = "None"
        import pdb; pdb.set_trace()
        return parse_gemini_output(input, output)
    except Exception as e:
        if not force:
            traceback.print_exc()
            raise e


def main(args: argparse.Namespace):
    # check for google api key
    # assert "GOOGLE_API_KEY" in os.environ

    # load dataset
    dataset = json.load(args.dataset.open("r"))
    print("found {:,} questions".format(len(dataset)))

    # load results
    results = []
    if args.output_path.exists():
        results = json.load(args.output_path.open())
        print("found {:,} existing results".format(len(results)))
    completed = [item["question_id"] for item in results]
    
    # model
    model_path = "/home/tianqi/.cache/huggingface/hub/models--LanguageBind--Video-LLaVA-7B-hf"
    model_base = None
    model_name = "Video-LLaVA-7B"
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base, model_name)
    model = model.to('cuda:1')
    import pdb; pdb.set_trace()
    
    # model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", device_map="auto")
    # processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

    # process data
    for idx, item in enumerate(tqdm.tqdm(dataset)):
        if args.dry_run and idx >= 5:
            break

        # skip completed questions
        question_id = item["question_id"]
        if question_id in completed:
            continue  # skip existing

        # extract scene paths
        folder = args.frames_directory / item["episode_history"]
        frames = sorted(folder.glob("*-rgb.png"))
        indices = np.round(np.linspace(0, len(frames) - 1, args.num_frames)).astype(int)
        paths = [str(frames[i]) for i in indices]

        # generate answer
        question = item["question"]
        answer = ask_question(
            frame_paths=paths,
            question=question,
            image_size=args.image_size,
            google_model=model,
            force=args.force,
            processor=processor,
        )

        # store results
        results.append({"question_id": question_id, "answer": answer})
        json.dump(results, args.output_path.open("w"), indent=2)

    # save at end (redundant)
    json.dump(results, args.output_path.open("w"), indent=2)
    print("saving {:,} answers".format(len(results)))


if __name__ == "__main__":
    main(parse_args())
