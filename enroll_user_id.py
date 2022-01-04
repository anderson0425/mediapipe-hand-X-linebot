#登記line user id
from __future__ import unicode_literals
from os import path

import sys
from random import randint
import json
import threading
import configparser

from flask import Flask, request, abort, make_response
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageSendMessage

# initialize a Flask object
app = Flask(__name__)

# get config values from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# initialize objects related to line-bot-sdk
line_bot_api = LineBotApi(config.get('line-bot', 'channel_access_token'), timeout=3000)
handler = WebhookHandler(config.get('line-bot', 'channel_secret'))

# pre-defined JSON file path of user_id list
# will be create automatically if the file is not exist or the data structure is not a list
USER_LIST_FILE = './user.json'


#這個網頁路徑會記錄line用戶傳給line-bot的訊息
@app.route('/callback', methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    print(body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'


#這是當被觸發時，line-bot會怎麼傳訊息給line用戶的程式。
@handler.add(MessageEvent, message=TextMessage)
def message_text(event):
    msg = event.message.text
    print(msg)

    # enroll as a user
    if event.message.text == 'enroll':
        print('got a \"enroll\" message')

        # append current user id into user_list file
        with open(USER_LIST_FILE, 'r', encoding='utf-8') as file:
            user_list = json.load(file)

        user_list.append(event.source.user_id)
        user_list = list(set(user_list))

        with open(USER_LIST_FILE, 'w', encoding='utf-8') as file:
            json.dump(user_list, file, ensure_ascii=False, indent=4)

        # reply success message
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text='success'))

def ss():
    s=[]

if __name__ == '__main__':
    try:
        #print("hi 1")
        with open(USER_LIST_FILE, 'r', encoding='utf-8') as file:
            user_list = json.load(file)
        if type(user_list) != list:
            raise TypeError()
    except:
        with open(USER_LIST_FILE, 'w', encoding='utf-8') as file:
            json.dump(list(), file, ensure_ascii=False, indent=4)
    finally:
        threading.Thread(target=ss).start()
        app.run(debug=False, port=5000)