import os
import requests
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("BOT_TOKEN")
user_id = os.getenv("USER_ID")

url = f"https://api.telegram.org/bot{token}"

message = "sup"

params = {"chat_id": user_id, "text": message}

r = requests.get(url + "/sendMessage", params=params)


print(r)