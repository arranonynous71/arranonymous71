# Revisiting Intermediate-Layer Matching Knowledge Distillation: Layer-Selection Strategy Doesn’t Matter (Much)

## Classification 

### Environment Details 
- Python: `3.9.19` 
- Huggingface: `4.1.0`
- PyTorch: `2.3.0`
- PyTorch-Lightning: `1.0.4`

### Steps to Reproduce Results
1. Download dataset or model if necessary (i.e. if compute node is not connected to the internet): 
    ```bash 
    python3 download-dataset-for-cc.py 
    python3 model-download-for-cc.py
    ```
    Before any training, you should also consider making a `runs` and `models` directory to contain experiments and hold onto models. 

2. Finetune teacher: 
    ``` bash 
    bash scripts/run_finetune_teacher.sh
    ```
    > Note: You might want to edit the paths in the `.sh` script
3. Distil student models from teacher:
    - Run any experiment in any following folder under `scripts/`: 
        - `all-to-one`
        - `depth`
        - `forward`
        - `reverse`
        - `mle`
        - `random-shuffle`
        - `width`
    - Be sure to run them from the `classification` directory. For example: 
        ```bash 
        bash scripts/all-to-one/mnli-all-one-random.sh
        ```
4. Evaluate the model. Make sure you use the best model saved under `best_tfmr` of each experiment.

    ```bash 

    bash scripts/eval.sh
    ```
5. Distance and Angles Analysis: 
    ```bash 
    bash scripts/run_dist_analysis.sh
    ```

### Results on MNLI, QQP, QNLI, SST2 

| Model   |                       | Layer Selection | MNLI-m/mm Acc | QQP Acc/F1 | QNLI Acc | SST-2 Acc |
|---------|-----------------------|-----------------|:-------------:|:----------:|:--------:|:---------:|
| Teacher | Previous work         | –               |   84.6/83.4   |   – /71.2  |   90.5   |    93.5   |
|         | Our replication       | –               |   84.5/84.1   |  89.0/71.4 |   90.8   |    93.1   |
| Student | Randomly Initialized  | None            |   63.2/63.6   |  81.5/56.4 |   61.2   |    81.1   |
|         |                       | Forward         |   72.5/72.0   |  83.9/61.3 |   64.7   |    85.1   |
|         |                       | Reverse         |   69.3/68.9   |  84.3/61.8 |   65.2   |    83.3   |
|         |                       | All-to-one      |   74.0/73.8   |  83.4/60.2 |   65.0   |    85.4   |
|         |                       | Random Matching |   71.2/71.2   | 82.4/58.8  |   64.4   |   82.9    |
|         | Weights copied        | None            |   77.4/76.5   |  87.6/67.1 |   81.2   |    88.7   |
|         |                       | Forward         |   79.7/78.8   |  88.2/69.1 |   83.8   |    92.3   |
|         |                       | Reverse         |   79.2/78.2   |  88.1/68.3 |   83.2   |    90.0   |
|         |                       | All-to-one      |   79.4/78.7   |  87.6/68.6 |   82.8   |    91.4   |




## Data to Text / Machine Translation 

### Environment Details 
- Python: `3.9.19` 
- Huggingface: `4.1.0`
- PyTorch: `2.3.0`
- PyTorch-Lightning: `1.0.4`

### Steps to Reproduce Results

Make sure you are in the respective directories: `dart` for Data to Text or `t5mt` for Machine Translation.

1. Download dataset:
    - DART: [a google drive link that will be revealed once anonymity is lifted](a_link)
    - WMT16 En--Ro: [a google drive link that will be revealed once anonymity is lifted](a_link)

    Before any training, you should also consider making a `runs` and `models` directory to contain experiments and hold onto models.

2. Finetune teacher: 

    ```bash

    bash scripts/run_finetune_teacher.sh 
    ```
3. Distil student models from teacher:
    - Run any experiment in any following folder under `scripts/`: 
        - `all-to-one`
        - `width`
        - `forward`
        - `reverse`
        - `mle`
        - `random-shuffle`
    - Be sure to run them from the `classification` directory. For example: 
        ```bash 
        bash scripts/all-to-one/all-one-3l-random.sh 
        ```
4. Evaluate the model. Make sure you use the best model saved under `best_tfmr` of each experiment.
    - DART: 
        1.  first run `run_eval.sh`, which generates predictions stored in `res.out`. 
        2.  Then feed `res.out` into `run_eval_on_dart.sh`
    - WMT16 En--Ro: 
        1.  run `run_eval.sh`
5. Distance and Angles Analysis: 
    - WMT16 En--Ro: 
        ```bash 
        bash scripts/run_dist_analysis.sh
        ```  
### Results on DART and WMT16 En-Ro

| Model   |                       | Layer Selection | DART BLEU | WMT-16 En-Ro BLEU |
|---------|-----------------------|-----------------|:---------:|:-----------------:|
| Teacher | Previous work         | –               | 48.56     | 25.82             |
|         | Our replication       | –               | 48.80     | 25.90             |
| Student | Randomly Initialized  | None            | 38.76     | 8.02              |
|         |                       | Forward         | 32.64     | 18.13             |
|         |                       | Reverse         | 33.12     | 17.15             |
|         |                       | All-to-one      | 33.86     | 17.16             |
|         |                       | Random Matching | 32.67     | 16.70             |
|         | Weights copied        | None            | 46.32     | 22.36             |
|         |                       | Forward         | 47.94     | 22.65             |
|         |                       | Reverse         | 48.45     | 21.57             |
|         |                       | All-to-one      | 47.10     | 21.89             |

## Challenging tasks with LLM


### Environment Details 
- Python: `3.11`
- Huggingface: `4.51.0`
- PyTorch: `2.7.1`

### Installation

Go to the `llm` folder, then install the dependencies. 

```bash 
# create env
python -m venv env 

# activate env
source env/bin/activate 

# install general requirements
pip install -r requirements.txt 

# install lm-evaluation-harness 
cd lm-evaluation-harness 
pip install -e .
```

### Steps to Reproduce Results

1. (Optional) If you are on an offline compute node, you may want to pre-download your model and datasets.

    ```bash
    # download hellaswag dataset. you may also download 'coqa' (CommonsenseQA)
    python download_dataset.py hellaswag

    # download Qwen3-8B Causal LM. Saves to <save_dir>/Qwen3/Qwen3-8B. 
    # If save_dir is not specified, saves to current working directory.
    python download_model.py Qwen/Qwen3-8B --dtype bfloat16 --save_to <save_dir>
    ```
    > Note: If you pre-download your models, pass the `--force_load_local_dataset` and `--local_dataset_dir` args to `finetune.py` and `distillation.py`

2. Run finetuning script to obtain finetuned Teacher model. By default the scritps will save the checkpoints to `runs` directory.

    ```bash
    bash scripts/hellaswag/finetune-teacher-hellaswag.sh
    ```
3. Merge teacher adapters to unify the teacher model. 

    ```bash 
    # Example: if your teacher model is `runs/teacher`
    python merge_adapters.py \
        --base_model Qwen/Qwen3-8B \
        --adapter runs/teacher/best_tfmr \
        --save_as runs/teacher
    ```
4. Run distillation experiments. Available distillation experiments include: 
    - `forward`: Forward Matching
    - `reverse`: Reverse Matching
    - `all_one`: All-to-One Matching
    - `shuffle`: Out-of-Order Random Matching
    - `none`: No Matching

    For example: 

    ```bash 
    # Example: Runs Forward Matching Distillation experiment on Hellaswag.
    # Be sure to edit this file and specify the finetuned teacher model!
    bash scripts/hellaswag/distill-forward-hellaswag.sh
    ```

5. Evaluate the student model:
    1. Make sure to merge the student adapters. Use the `--weight_copy` argument to ensure the student is initialized with the correct number of layers

        ```bash 
        python merge_adapters.py \
            --base_model Qwen/Qwen3-8B \
            --weight_copy \
            --adapter runs/<student_path>/best_tfmr \
            --save_as runs/<student_path>
        ```

    2. Run `lm_eval`: 

        ```bash 
        bash scripts/run_lm_eval.sh 
        ```

### Results on HellaSwag and CommonsenseQA

| Models  | Layer Matching      | # | HellaSwag Acc | CommonsenseQA Exact Match |
|---------|---------------------|:-:|:-------------:|:-------------------------:|
| Teacher | --                  | 1 |     63.11     |           75.02           |
| Student | None                | 2 |     35.16     |           33.73           |
|         | Forward             | 3 |     37.85     |           35.40           |
|         | Reverse             | 4 |     35.01     |           37.67           |
|         | All-to-one          | 5 |     34.99     |           37.35           |
|         | Out-of-order random | 6 |     35.47     |           34.80           |
