"""Compare HuggingFace and vLLM generation for FlexOlmoNoQKNormPrenormShared.

Usage:
    python tests/models/test_flex_olmo_noqknorm_prenorm_shared.py \
        --model /path/to/hf/checkpoint

Requires the custom transformers fork with the
flex_olmo_noqknorm_prenorm_shared model type installed.
"""

import argparse

import torch


PROMPTS = [
    "The capital of France is",
    "In a distant galaxy, scientists discovered",
    "The quick brown fox jumps over the",
    "Machine learning models are trained by",
]


def generate_with_hf(model_path: str, max_new_tokens: int):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=== HuggingFace generation ===")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    results = []
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        # Decode only the generated part
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        results.append(text)
        print(f"  Prompt: {prompt!r}")
        print(f"  Output: {text!r}\n")

    # Free memory
    del model
    torch.cuda.empty_cache()
    return results


def generate_with_vllm(model_path: str, max_new_tokens: int):
    from vllm import LLM, SamplingParams

    print("=== vLLM generation ===")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
    )

    outputs = llm.generate(PROMPTS, sampling_params)
    results = []
    for output in outputs:
        text = output.outputs[0].text
        results.append(text)
        print(f"  Prompt: {output.prompt!r}")
        print(f"  Output: {text!r}\n")

    del llm
    torch.cuda.empty_cache()
    return results


def compare(hf_results: list[str], vllm_results: list[str]):
    print("=== Comparison ===")
    all_match = True
    for i, (hf_text, vllm_text) in enumerate(
        zip(hf_results, vllm_results)
    ):
        match = hf_text == vllm_text
        status = "MATCH" if match else "MISMATCH"
        print(f"  Prompt {i}: {status}")
        if not match:
            all_match = False
            print(f"    HF:   {hf_text!r}")
            print(f"    vLLM: {vllm_text!r}")
    print()
    if all_match:
        print("All outputs match!")
    else:
        print("Some outputs differ — check for numerical precision issues.")


def main():
    parser = argparse.ArgumentParser(
        description="Compare HF vs vLLM generation for FlexMoE"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to HF checkpoint"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=20,
        help="Max tokens to generate per prompt",
    )
    parser.add_argument(
        "--hf-only", action="store_true", help="Only run HF generation"
    )
    parser.add_argument(
        "--vllm-only", action="store_true", help="Only run vLLM generation"
    )
    args = parser.parse_args()

    hf_results = None
    vllm_results = None

    if not args.vllm_only:
        hf_results = generate_with_hf(args.model, args.max_new_tokens)

    if not args.hf_only:
        vllm_results = generate_with_vllm(args.model, args.max_new_tokens)

    if hf_results is not None and vllm_results is not None:
        compare(hf_results, vllm_results)


if __name__ == "__main__":
    main()
