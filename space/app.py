import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "Bibidh/civicLens-llama3.2-3b-nepali-legal-merged"

SYSTEM_PROMPT = (
    "You are CivicLens, a legal assistant specialized in Nepal's laws, "
    "constitution, and governance documents. Answer questions accurately, "
    "cite your sources, and respond in the same language as the question. "
    "If you don't know, say so."
)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,  # CPU requires float32
    device_map="cpu",
    low_cpu_mem_usage=True,
)
model.eval()
print("Model loaded.")


@torch.inference_mode()
def respond(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user",      "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    prompt  = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs  = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


demo = gr.ChatInterface(
    fn=respond,
    title="CivicLens Nepal",
    description="Ask questions about Nepal's laws, constitution, and governance documents. Answers are provided in the language of your question.",
    examples=[
        "What are the fundamental rights guaranteed by the Constitution of Nepal?",
        "नेपालको संविधानमा मौलिक हकहरू के के छन्?",
        "What is the process for impeachment of the President of Nepal?",
    ],
    theme=gr.themes.Soft(),
)

demo.launch()