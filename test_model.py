import torch
from transformers import AutoProcessor, Idefics2ForConditionalGeneration
from datasets import load_dataset
from PIL import Image
import io

model = Idefics2ForConditionalGeneration.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")

data = load_dataset("ahmed-masry/ChartQA", split="test")
example = data[0]

question = example['query']
answer = example['label']
image = Image.open(io.BytesIO(example['image']))

print(f"Question: {question}")
print(f"Ground truth: {answer}")

messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=text, images=[image], return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

output = model.generate(**inputs, max_new_tokens=100)
response = processor.decode(output[0], skip_special_tokens=True)

answer_text = response.split("Assistant:")[-1].strip()

print(f"Model answer: {answer_text}")
