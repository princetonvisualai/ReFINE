# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py

import re
import string
from collections import Counter
from typing import Union

def compute_score(response: str, answers: list[str], **kwargs):
    response = response.split("\n")[0]
    response = postprocess_response(response)
    output = 0.0
    for ground_truth in answers:
        score = 1.0 if ground_truth.lower() in response.lower() else 0.0
        output = max(score, output)

    #response = response.split("\n")[0]
    #output = get_qa_f1_score(response, answers)
    return output

def postprocess_response(response: str):
    response = response.strip()   
    np_pattern = re.compile(r"[\x00-\x1f]")
    response = np_pattern.sub("\n", response).strip()
    return response

def get_qa_f1_score(response: str, answers: list[str]):
    output = 0.0
    response = response.strip()
    for ground_truth in answers:
        score = qa_f1_score(response, ground_truth)
        output = max(score, output)
    return output

def qa_f1_score(response: str, answer: str):
    normalized_response = normalize_answer(response)
    normalized_answer = normalize_answer(answer)

    response_tokens = normalized_response.split()
    answer_tokens = normalized_answer.split()
    return f1_score(response_tokens, answer_tokens)

def f1_score(response: Union[str, list], answer: Union[str, list]):
    common = Counter(response) & Counter(answer)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(response)
    recall = 1.0 * num_same / len(answer)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


