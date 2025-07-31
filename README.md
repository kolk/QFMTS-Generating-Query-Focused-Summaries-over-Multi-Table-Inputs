# QFMTS: Generating Query Focused Summaries over Multi-Table Inputs

**Datasets**
  - Download the dataset at [QFMTS dataset](https://huggingface.co/datasets/vaishali/qfmts_query_focused_multitab_summarization)
    
Alternatively, load the dataset from huggingface hub:
```
from datasets import load_dataset
qfmts_dataset = load_dataset("vaishali/qfmts_query_focused_multitab_summarization")
training_set, validation_set, test_set = qfmts_dataset['training'], qfmts_dataset['validation'], qfmts_dataset['test']
```


**Model Checkpoints**
  -  Download [`BART-base-FT`](https://huggingface.co/vaishali/) 
  -  Download [`BART-large-FT`](https://huggingface.co/vaishali/) 
  -  Download [`TAPEX-FT`](https://huggingface.co/vaishali/) 
    - Download [`MultiTab-FT`](https://huggingface.co/vaishali/)
    - Download [`OmniTab-FT`](https://huggingface.co/vaishali/)
   - Download [`Llama-2-FT`](https://huggingface.co/vaishali/)
   - Download [`Reason-then-Summ-llama2`](https://huggingface.co/vaishali/)

**Loading encoder-decoder Model Checkpoints**

  - *BART / MultiTab / TAPEX / OmniTab* 
```
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
```

- *Llama*
```
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
model_name = LLAMA-2-7b_DIRECTORY
adapters_name = "checkpoints/"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map={"": 0}
    )
    model = PeftModel.from_pretrained(model, adapters_name)
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
```
**Training Process**

Arguments for encoder-decoder training:
```

python tableqa/train.py --pretrained_language_model "vaishali/multitabqa-base" \
                --learning_rate 1e-4 \
                --train_batch_size 2 \
                --eval_batch_size 2 \
                --gradient_accumulation_steps 64 \
                --num_train_epochs 8 \
                --use_multiprocessing False \
                --num_workers 2 \
                --decoder_max_length 1024 \
                --seed 42 \
                --decoder_max_length 1024 \
                --language "bn" \
                --output_dir "experiments/multitabqa-ft" 

```

Arguments for Llama training:
```
python tableqa/train.py --pretrained_language_model "llama-2-7b-hf" \
                --learning_rate 1e-4 \
                --train_batch_size 8 \
                --eval_batch_size 8 \
                --gradient_accumulation_steps 4 \
                --num_train_epochs 5 \
                --save_total_limit 50 \
                --seed 1234 \
                --warmup_ratio 0.04 \
                --use_multiprocessing False \
                --num_workers 2 \
                --decoder_max_length 1024 \
                --local_rank -1 \
                --language "bn" \
                --dataset "banglaTabQA" \
                --load_in8_bit \
                --r 8 \
                --lora_alpha 16  \
                --output_dir "experiments/llama_ft" 
```


Arguments for evaluation:
```
python tableqa/evaluate.py --pretrained_model_name "experiments/multitabqa-ft" \
                --batch_size 2 --generation_max_length 1024 \
                --dataset_split "test_set" \
                --predictions_save_path "experiments/predictions/multitabqa-ft_test.jsonl" 
```

Please cite our work if you use our code or datasets:
```
@misc{zhang2024qfmtsgeneratingqueryfocusedsummaries,
      title={QFMTS: Generating Query-Focused Summaries over Multi-Table Inputs}, 
      author={Weijia Zhang and Vaishali Pal and Jia-Hong Huang and Evangelos Kanoulas and Maarten de Rijke},
      year={2024},
      eprint={2405.05109},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.05109}, 
}
```
