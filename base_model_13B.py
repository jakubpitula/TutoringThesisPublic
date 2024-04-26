import torch
import transformers

from transformers import LlamaForCausalLM, LlamaTokenizer
from gen_preprocess import gen_preprocess
from alive_progress import alive_bar

model_dir = "../llama/llama-2-13b-chat-hf"
model = LlamaForCausalLM.from_pretrained(model_dir, low_cpu_mem_usage=True)
tokenizer = LlamaTokenizer.from_pretrained(model_dir)

preprocess_results = gen_preprocess(split=0.8, sample_len=9)

prompts = preprocess_results['prompts']
samples = preprocess_results['samples']
max_len = preprocess_results['max_gen_length']

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

with alive_bar(len(prompts), force_tty=True) as bar:
    for prompt, sample in zip(prompts, samples):
        print(sample['context'][0]['turn'])
        # seems like return_full_text=False and pad_token_id=tokenizer.pad_token_id do the trick
        sequences = pipeline(
            prompt,
            do_sample=True,
            top_k=50,
            num_return_sequences=1,
            max_new_tokens=512,
            return_full_text=False,
            pad_token_id=tokenizer.pad_token_id
        )

        with open('base_model_out.txt', 'a') as file:
            for turn in sample['context']:
                file.write(f"{turn['speaker'].upper()}: {turn['turn']}\n")

            file.write(f"\nORIGINAL SCAFFOLDING: {sample['scaffolding']}\n")

            for seq in sequences:
                file.write(f"GENERATED SCAFFOLDING: {seq['generated_text']}\n")

            file.write("\n==================================\n")
        bar()

