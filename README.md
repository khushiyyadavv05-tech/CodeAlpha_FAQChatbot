# CodeAlpha_FAQChatbot

FAQ chatbot built for the **CodeAlpha Artificial Intelligence Internship (Task 2)**.
It answers questions about the CodeAlpha internship (tasks, deadlines, submission, certificates, support).

## How it works

1. **Collect FAQs**: questions and answers are stored in `faqs.json`.
2. **Preprocess text**: lowercase, remove punctuation, tokenize, remove stopwords, stemming.
3. **Vectorize**: FAQs are converted to TF-IDF vectors (unigrams + bigrams).
4. **Match**: the user's question is compared with every FAQ using **cosine similarity**.
5. **Respond**: the best match's answer is shown. If the match is too weak, a fallback message is returned.

## Project structure

```
CodeAlpha_FAQChatbot/
├── app.py              # Flask web server (chat UI)
├── chatbot.py          # NLP + matching engine (also runs in terminal)
├── faqs.json           # FAQ data (edit this to change the topic)
├── templates/
│   └── index.html      # Chat interface
├── requirements.txt
└── README.md
```

## Setup and run

```bash
pip install -r requirements.txt

# Web chat UI
python app.py
# open http://127.0.0.1:5000

# Or terminal chat
python chatbot.py
```

NLTK is optional: if installed, its `PorterStemmer` is used, otherwise a built-in stemmer is used.

## Customize

Edit `faqs.json`. Each entry looks like:

```json
{
  "question": "What is the submission deadline?",
  "alt": ["last date to submit", "due date"],
  "answer": "The deadline is 20th December 2026, 11:59 PM."
}
```

`alt` is optional and holds alternate phrasings that improve matching.

## Tech stack

Python, Flask, scikit-learn (TF-IDF, cosine similarity), NLTK (optional)
