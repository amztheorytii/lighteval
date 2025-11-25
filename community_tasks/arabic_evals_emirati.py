# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.
"""
import random
import re
from typing import Any, Dict, List, Optional, Union

from lighteval.metrics.metrics import Metric, MetricCategory, Metrics
from lighteval.metrics.utils.metric_utils import MetricUseCase
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# fmt: off
LETTER_INDICES_AR = ["أ", "ب", "ج", "د", "هـ", "و", "ز", "ح", "ط", "ي", "ك", "ل", "م", "ن", "س", "ع", "ف", "ص", "ق", "ر", "ش", "ت", "ث", "خ", "ذ", "ض", "ظ", "غ"]


# Emirati SYN Bench ##
EMIRATI_SUBSET = ["emirati_syn_bench"]

def emirati_syn_bench_pfn(line, task_name: str = None):
    question = line["query"]
    answer_index = int(line["label"])
    allowed_keys = [f"sol{i}" for i in range(1, 5)]
    extracted_choices = [line[key] for key in allowed_keys if key in line]
    # choices = [str(i) for i in range(len(extracted_choices))]

    instruction = "الأسئلة التالية هي أسئلة متعددة الإختيارات مع الجواب الصحيح\n\n"
    query = f"{instruction}السؤال: {question}\n"

    for index, choice in enumerate(extracted_choices):
        query += f"{LETTER_INDICES_AR[index]}) {choice}\n"
    
    query += "الإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES_AR[:4],
        gold_index=answer_index,
        instruction=instruction,
    )


alsharh_emirati_bench_task = LightevalTaskConfig(
    name="alsharh_emirati_bench",
    prompt_function=emirati_syn_bench_pfn,
    suite=["community"],
    hf_subset=None,
    hf_repo="falcon-arabic/Alsharh-v1",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    metric=[Metrics.loglikelihood_acc_norm],
    version=0,
)


silbar_emirati_bench_task = LightevalTaskConfig(
    name="silbar_emirati_bench",
    prompt_function=emirati_syn_bench_pfn,
    suite=["community"],
    hf_subset=None,
    hf_repo="falcon-arabic/Silbar-v1",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    metric=[Metrics.loglikelihood_acc_norm],
    version=0,
)



TASKS_TABLE = (
    [alsharh_emirati_bench_task]
    + [silbar_emirati_bench_task]
)