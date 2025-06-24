import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime

# Initialize Lisa's AI model (DialoGPT for conversational responses)
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# File to store conversation history (persists in Hugging Face Space)
MEMORY_FILE = "lisa_memory.json"

# Load past conversations
def load_memory():
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"conversations": {}}

# Save new conversations
def save_memory(data):
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f)

# Lisa's personality
lisa_personality = """
You are Lisa, a friendly and helpful AI assistant.  
You remember past conversations and respond naturally.  
Your interests: technology, art, and science.  
Keep responses concise and human-like.  
"""

# Generate Lisa's response with memory
def chat(user_input, user_id="default"):
    memory = load_memory()
    
    # Get past 3 messages for context
    past_conversations = memory["conversations"].get(user_id, [])[-3:]
    context = "\n".join([f"User: {msg['user']}\nLisa: {msg['response']}" for msg in past_conversations])
    
    # Create prompt with personality + memory
    prompt = f"{lisa_personality}\n{context}\nUser: {user_input}\nLisa:"
    
    # Generate response
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs, 
        max_length=200,
        do_sample=True,
        temperature=0.7
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Lisa:")[-1].strip()
    
    # Update memory
    if user_id not in memory["conversations"]:
        memory["conversations"][user_id] = []
    
    memory["conversations"][user_id].append({
        "user": user_input,
        "response": response,
        "time": str(datetime.now())
    })
    
    save_memory(memory)
    return response

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Lisa - Your Personal AI Assistant")
    user_id = gr.Textbox(label="Your Name (for memory)", value="user_123")
    user_input = gr.Textbox(label="Talk to Lisa")
    output = gr.Textbox(label="Lisa's Response")
    submit = gr.Button("Send")
    submit.click(chat, inputs=[user_input, user_id], outputs=output)

demo.launch()
