"""Flask web UI for the FAQ chatbot. Run: python app.py  ->  http://127.0.0.1:5000"""

from flask import Flask, jsonify, render_template, request

from chatbot import FAQChatbot

app = Flask(__name__)
bot = FAQChatbot()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", ""))
    return jsonify(bot.get_response(message))


if __name__ == "__main__":
    app.run(debug=True)
