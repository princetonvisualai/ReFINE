# MIT License
#
# Copyright (c) 2023 THU-KEG & Zhipu AI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import re
import string
from collections import Counter
from typing import Union
from functools import partial

try:
    import jieba
    from fuzzywuzzy import fuzz
    from rouge import Rouge
except ImportError:
    raise ImportError(
        'Please install the required dependencies for this task with `pip install lm_eval["longbench"] or `pip install jieba fuzzywuzzy rouge`'
    )

# taken and slightly modified from https://github.com/THUDM/LongBench


def compute_score(response: str, answers: list[str], data_source: str):
    
    dataset2metric = {
        "narrativeqa": get_qa_f1_score,
        "qasper": get_qa_f1_score,
        "multifieldqa_en": get_qa_f1_score,
        "hotpotqa": get_qa_f1_score,
        "2wikimqa": get_qa_f1_score,
        "musique": get_qa_f1_score,
        "gov_report": get_rouge_score,
        "samsum": get_rouge_score,
        "qmsum": get_rouge_score,
        "multi_news": get_rouge_score,
        "trec": get_classification_score,
        "triviaqa": get_qa_f1_score,
        "lcc": get_code_sim_score,
        "repobench-p": get_code_sim_score,
    }
    if data_source not in dataset2metric:
        raise ValueError(f"Invalid data source: {data_source}")
    
    if data_source in ["trec", "triviaqa", "samsum", "lsht"]:
        response = response.lstrip('\n').split('\n')[0]
    elif data_source in ["gov_report", "samsum", "qmsum", "multi_news"]:
        pass
    elif data_source in ["lcc", "repobench-p"]:
        pass
    else: # single, multi doc qa
        response = response.split("\n")[0]
        

    output = dataset2metric[data_source](response, answers)
    return output
    
    
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

def get_code_sim_score(response: str, answers: list[str]):
    output = 0.0
    for ground_truth in answers:
        score = code_sim_score(response, ground_truth)
        output = max(score, output)
    return output

def code_sim_score(response: str, ground_truth: str):
    all_lines = response.lstrip("\n").split("\n")
    prediction = ""
    for line in all_lines:
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            response = line
            break
    return fuzz.ratio(response, ground_truth) / 100

def get_rouge_score(response: str, answers: list[str]):
    output = 0.0
    response = response.strip()
    for ground_truth in answers:
        score = rouge_score(response, ground_truth)
        output = max(score, output)
    return output

def rouge_score(response: str, answer: str) -> float:
    global rouge
    if "rouge" not in globals():
        rouge = Rouge()
    try:
        scores = rouge.get_scores([response], [answer], avg=True)
        # ruff: noqa
    except:
        return 0.0
    return scores["rouge-l"]["f"]

def get_classification_score(response: str, answers: list[str]):
    output = 0.0
    response = response.strip()
    for ground_truth in answers:
        score = classification_score(response, ground_truth)
        output = max(score, output)
    return output

def classification_score(response: str, ground_truth: str):
    em_match_list = []
    all_classes = [
        "Food", 
        "Date", 
        "Order, rank", 
        "Speed", 
        "Disease and medicine", 
        "Word with a special property", 
        "Abbreviation", 
        "Language", 
        "Letter like a-z", 
        "Other entity", 
        "Animal", 
        "Expression abbreviated", 
        "Price",
        "Techniques and method", 
        "Musical instrument", 
        "Mountain", 
        "Currency name", 
        "Event", 
        "Product", 
        "State", 
        "Individual", 
        "Organ of body", 
        "Reason", 
        "Manner of an action", 
        "City", 
        "Religion", 
        "Invention, book and other creative piece",
        "Distance, linear measure", 
        "Temperature", 
        "Postcode or other code", 
        "Size, area and volume", 
        "Sport",
        "Country", 
        "Other location", 
        "Lasting time of somethin", 
        "Equivalent term", 
        "Description of something", 
        "Weight", 
        "Vehicle", 
        "Color", 
        "Other number", 
        "Definition of something", 
        "Element and substance", 
        "Description of a person", 
        "Symbols and sign", 
        "Number of something", 
        "Plant", 
        "Percent, fraction", 
        "Group or organization of person", 
        "Title of a person"
    ]
    for class_name in all_classes:
        if class_name in response:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        score = 1.0 / len(em_match_list)
    else:
        score = 0.0
    return score

def get_recall_score(response: str, answers: list[str]):
    output = 0.0
    response = response.strip()
    for ground_truth in answers:
        score = recall_score(response, ground_truth)
        output = max(score, output)
    return output

def recall_score(response: str, answer: str) -> float:
    score = 0.0
    if response.lower() in answer.lower():
        score = 1.0
    return score

