# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import traceback
from typing import Optional

from openeqa.utils.openai_utils import (
    call_openai_api,
    prepare_openai_messages,
    set_openai_key,
)
from openeqa.utils.prompt_utils import load_prompt
from openeqa.utils.llama_utils import LLaMARunner, enable_full_determinism


def parse_score(output: str, tag: str = "Your mark:") -> str:
    if output.isdigit():
        return int(output)
    start_idx = output.find(tag)
    if start_idx == -1:
        raise ValueError("Invalid output string: {}".format(output))
    end_idx = output.find("\n", start_idx)
    if end_idx == -1:
        return int(output[start_idx:].replace(tag, "").strip())
    return int(output[start_idx:end_idx].replace(tag, "").strip())


def get_llm_match_score(
    question: str,
    answer: str,
    prediction: str,
    extra_answers: Optional[list] = None,
    openai_key: Optional[str] = None,
    openai_model: str = "gpt-4-1106-preview",
    openai_seed: int = 1234,
    openai_max_tokens: int = 32,
    openai_temperature: float = 0.2,
    verbose: bool = False,
    model: str = None,
):
    if prediction is None:
        return 0

    prompt_name = "mmbench" if extra_answers is None else "mmbench-extra"
    prompt = load_prompt(prompt_name)

    try:
        # set_openai_key(key=openai_key)
        messages = prepare_openai_messages(
            prompt.format(
                question=question,
                answer=answer,
                prediction=prediction,
                extra_answers=extra_answers,
            ),
        )
        output = call_openai_api(
            messages=messages,
            model=openai_model,
            seed=openai_seed,
            max_tokens=openai_max_tokens,
            temperature=openai_temperature,
            verbose=verbose,
        )
        # output = ask_question(model=model, question=messages[0]['content'])
        return parse_score(output)
    except Exception as e:
        traceback.print_exc()
        raise e

def ask_question(
    model, question: str, max_tokens: int = 128, temperature: float = 0.2
) -> Optional[str]:
    # prompt = load_prompt("blind-llm")
    # input = prompt.format(question=question)
    output = model(question, max_new_tokens=max_tokens, temperature=temperature)
    return parse_output(output)

def parse_output(output: str) -> str:
    start_idx = output.find("Your mark:")
    if start_idx == -1:
        raise ValueError("Invalid output string: {}".format(output))
    end_idx = output.find("\n", start_idx)
    if end_idx == -1:
        return output[start_idx:].replace("Your mark:", "").strip()
    return output[start_idx:end_idx].replace("Your mark:", "").strip()

if __name__ == "__main__":
    # example usage
    question = "What color is the rug?"
    answer = "tan with pink and blue"
    prediction = "brown with pink and blue"
    # load model
    model = LLaMARunner(
        "/g/data/hn98/models/llama/llama2-7b",
        load_in_8bit=False,
        use_fast_kernels=False,
    )
    score = get_llm_match_score(question, answer, prediction, model=model)
    print(score)
    print("*" * 40)
    print("example question:    {}".format(question))
    print("ground-truth answer: {}".format(answer))
    print("predicted answer:    {}".format(prediction))
    print("llm-match score:     {}".format(score))
    print("*" * 40)