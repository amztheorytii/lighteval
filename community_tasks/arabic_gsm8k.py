from lighteval.metrics.metrics import  Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def gsm8k_pfn(line, task_name: str = None):
    # Has special analysis in metric for number decomposition
    return Doc(
        task_name=task_name,
        query=f"السؤال: {line['question']}\nالجواب:",
        choices=[f" {line['answer']}"],
        gold_index=0,
    )
    
GSM8K = LightevalTaskConfig(
    name="gsm8k_arabic",
    prompt_function=gsm8k_pfn,
    suite=["community"],
    hf_repo="Omartificial-Intelligence-Space/Arabic-gsm8k",
    hf_subset="",
    hf_avail_splits=["main_test","main_train"],
    evaluation_splits=["main_test"],
    few_shots_split=["main_train"],
    metric=[Metrics.quasi_exact_match_gsm8k],
    generation_size=1048,
    stop_sequence=None,
)

TASKS_TABLE = [GSM8K]
