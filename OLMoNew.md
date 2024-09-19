
## vllm for Peteish

new vllm olmo version: [olmo_new.py](https://github.com/AkshitaB/vllm/blob/main/vllm/model_executor/models/olmo_new.py)

### How to run

- Install vllm from my version (warning: installing from source takes AGES; upwards of 2 hours).
- convert peteish checkpoint to hf\_olmo style checkpoint (I tested with `/net/nfs.cirrascale/allennlp/akshitab/model-checkpoints/peteish7/step11931-unsharded-hf` — from [peteish7-anneal-from-928646-50B-nowup-dclm07-flan](https://us-east-1.console.aws.amazon.com/s3/buckets/ai2-llm?prefix=checkpoints/OLMo-medium/peteish7-anneal-from-928646-50B-nowup-dclm07-flan/))
- Then run vllm as usual (make sure to import hf_olmo)

```python
from hf_olmo import *
s = SamplingParams(temperature=0.0)
llm = LLM(model=path, trust_remote_code=True, gpu_memory_utilization=0.90)

set_random_seed(0)
vllm_out = llm.generate([prompt], sampling_params=s)
outputs["vllm"] = vllm_out[0].outputs[0].text
```

### Things to keep in mind

- This is a barebones implementation which works exactly with peteish config (things like RMSNorm are hardcoded to keep implementation simple). I have not added clean if-else statements for different norm types, etc. A non-peteish model will likely not produce the right results.

- Their RMSNorm cuda kernal implementation causes some discrepancies (still coherent outputs, but different than hf), so I use the `forward_native` call, which is native pytorch. This is slightly slower, so if you’re ok with different outputs than hf, swap out the native calls for regular calls here: https://github.com/AkshitaB/vllm/blob/main/vllm/model_executor/models/olmo_new.py#L144

### Benchmarking

- Run benchmarking script:

```python
export DATAP=/net/nfs.cirrascale/allennlp/akshitab/eval_data/ShareGPT_V3_unfiltered_cleaned_split.json
python benchmarks/benchmark_throughput.py --backend vllm --dataset $DATAP --model /net/nfs.cirrascale/allennlp/akshitab/model-checkpoints/peteish7/step11931-unsharded-hf
```

  Input: 1000 prompts from sharegpt

  | Run | Throughput | Clocktime |
  | --- | --- | --- |
  | peteish-vllm-regular-norm | Throughput: 13.00 requests/s, 5384.33 tokens/s | ~ 1 min |
  | peteish-vllm-native-norm (current implementation) | Throughput: 11.99 requests/s, 4968.23 tokens/s | ~ 1 min |
  | non-vllm hf baseline | was too slow to run full benchmark | > 1.5 hours (estimated) |
