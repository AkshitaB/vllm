
## vllm for Peteish

(https://github.com/AkshitaB/vllm)

- Install vllm from my version (warning: installing from source takes AGES; upwards of 2 hours).
- convert peteish checkpoint to hf\_olmo style checkpoint (I tested with `/net/nfs.cirrascale/allennlp/akshitab/model-checkpoints/peteish7/step11931-unsharded-hf` — from [peteish7-anneal-from-928646-50B-nowup-dclm07-flan](https://us-east-1.console.aws.amazon.com/s3/buckets/ai2-llm?prefix=checkpoints/OLMo-medium/peteish7-anneal-from-928646-50B-nowup-dclm07-flan/))
- Then run vllm as usual (make sure to import hf\_olmo)

```python
from hf_olmo import *
s = SamplingParams(temperature=0.0)
llm = LLM(model=path, trust_remote_code=True, gpu_memory_utilization=0.90)

set_random_seed(0)
vllm_out = llm.generate([prompt], sampling_params=s)
outputs["vllm"] = vllm_out[0].outputs[0].text
```

- Note: Their RMSNorm cuda kernal implementation causes some discrepancies (still coherent outputs, but different than hf), so I use the `forward_native` call, which is native pytorch. This is slightly slower, so if you’re ok with different outputs than hf, swap out the native calls for regular calls here: https://github.com/AkshitaB/vllm/blob/main/vllm/model\_executor/models/olmo\_new.py#L144
- Run benchmarking script:

```python
export DATAP=/net/nfs.cirrascale/allennlp/akshitab/eval_data/ShareGPT_V3_unfiltered_cleaned_split.json
python benchmarks/benchmark_throughput.py --backend vllm --dataset $DATAP --model /net/nfs.cirrascale/allennlp/akshitab/model-checkpoints/peteish7/step11931-unsharded-hf
```

- Results:
    
    peteish-vllm-regular-norm: 
    
    ```python
    Processed prompts: 100%|████████████████████████| 1000/1000 [01:21<00:00, 12.32it/s, est. speed input: 2641.08 toks/s, output: 2463.60 toks/s]
    Throughput: 12.23 requests/s, 5067.30 tokens/s
    ```
    
    peteish-vllm-native-norm (current implementation): 
    
    ```python
    Processed prompts: 100%|████████████████████████| 1000/1000 [01:16<00:00, 13.15it/s, est. speed input: 2819.28 toks/s, output: 2629.82 toks/s]
    Throughput: 13.05 requests/s, 5406.68 tokens/s
    ```
