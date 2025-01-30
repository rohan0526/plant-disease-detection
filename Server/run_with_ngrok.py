from flask import Flask
from pyngrok import ngrok
import requests
import time

app = Flask(__name__)

# Discord webhook URL
DISCORD_WEBHOOK_URL = '  # Replace with your actual webhook URL

def send_discord_message(message):
    data = {
        "content": message
    }
    response = requests.post(DISCORD_WEBHOOK_URL, json=data)

    if response.status_code == 204:
        print("Message sent to Discord successfully!")
    else:
        print(f"Failed to send message: {response.status_code}, {response.text}")

def start_ngrok_tunnel():
    print("Starting ngrok tunnel...")
    public_url = ngrok.connect(5000)  # Change the port if needed
    print(f"Ngrok tunnel available at: {public_url}")

    # Send the public URL via Discord
    send_discord_message(f'Hey guys! Your Ngrok Tunnel for Plant Disease Detection backend is ready.\nVisit: {public_url}\n\n For Potato Plant prediction use /predictPotato\n\n For Tomato Plant prediction use /predictTomato')

if __name__ == "__main__":
    start_ngrok_tunnel()
    try:
        while True:
            time.sleep(1)  # Keep the script running
    except KeyboardInterrupt:
        print("Ngrok tunnel is closing.")
        ngrok.disconnect()