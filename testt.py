import torch
from transformers import AutoTokenizer, TextStreamer, GenerationConfig
from attention_sinks import AutoModelForCausalLM


# model_id = "meta-llama/Llama-2-7b-hf"
# model_id = "mistralai/Mistral-7B-v0.1"
model_id = "mosaicml/mpt-7b"
# model_id = "tiiuae/falcon-7b"
# model_id = "EleutherAI/pythia-6.9b-deduped"
# Note: instruct or chat models also work.

# Load the chosen model and corresponding tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # for efficiency:
    device_map="auto",
    torch_dtype=torch.float16,
    # `attention_sinks`-specific arguments:
    attention_sink_size=4,
    attention_sink_window_size=252, # <- Low for the sake of faster generation
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Our input text
text = "Vaswani et al. (2017) introduced the Transformers"

# Encode the text
input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    # A TextStreamer prints tokens as they're being generated
    streamer = TextStreamer(tokenizer)
    generated_tokens = model.generate(
        input_ids,
        generation_config=GenerationConfig(
            # use_cache=True is required, the rest can be changed up.
            use_cache=True,
            min_new_tokens=100_000,
            max_new_tokens=1_000_000,
            penalty_alpha=0.6,
            top_k=5,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        ),
        streamer=streamer,
    )
    # Decode the final generated text
    output_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)