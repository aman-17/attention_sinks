import torch
from transformers import AutoTokenizer, TextStreamer, GenerationConfig
from attention_sinks import AutoModelForCausalLM

model_id = "allenai/OLMo-2-1124-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    attention_sink_size=4,
    attention_sink_window_size=252, # <- Low for the sake of faster generation
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id

text = """Write a comprehensive essay about the future of space exploration, including technological developments, human missions to Mars, establishing permanent colonies, and the eventual journey to other star systems. Provide detailed explanations and realistic timelines.
Introduction:
The next century of space exploration promises to be the most exciting period in human history. As we stand on the threshold of becoming a truly spacefaring civilization, we must examine the technological, economic, and social factors that will shape our expansion beyond Earth."""

input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    streamer = TextStreamer(tokenizer)
    generated_tokens = model.generate(
        input_ids,
        generation_config=GenerationConfig(
            use_cache=True,
            min_new_tokens=100_000,
            max_new_tokens=1_000_000,
            penalty_alpha=0.6,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
            top_k=5,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        ),
        streamer=streamer,
    )
    output_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
