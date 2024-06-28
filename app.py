
from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import os

# Initialize the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

app = Flask(__name__)

# Path to the uploaded CSV file
csv_file_path = "C:/Users/admin/OneDrive/Desktop/my_bot/Your Career Aspirations of GenZ.csv"

# Check if the CSV file exists, if not create it with appropriate columns
if not os.path.exists(csv_file_path):
    df = pd.DataFrame(columns=["user_input", "bot_response"])
    df.to_csv(csv_file_path, index=False)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    response = get_chat_response(input)
    return response

def get_chat_response(text):
    chat_history_ids = None

    # Let's chat for 5 lines
    for step in range(5):
        # Encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

        # Append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # Generate a response while limiting the total chat history to 1000 tokens
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # Get the response from the bot
        bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        # Save the user input and bot response to the CSV file
        save_to_csv(text, bot_response)

        # Return the bot's response
        return bot_response

def save_to_csv(user_input, bot_response):
    # Load existing CSV file
    df = pd.read_csv(csv_file_path)

    # Create a new entry
    new_entry = pd.DataFrame({"user_input": [user_input], "bot_response": [bot_response]})

    # Append the new entry to the DataFrame and save back to the CSV file
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(csv_file_path, index=False)

if __name__ == '__main__':
    app.run()
