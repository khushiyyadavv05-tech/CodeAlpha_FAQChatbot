"""
FAQ Chatbot engine (CodeAlpha AI Internship - Task 2)

Pipeline:
  1. Load FAQs (question + answer) from faqs.json
  2. Preprocess text: lowercase -> remove punctuation -> tokenize
     -> remove stopwords -> stem
  3. Convert FAQs to TF-IDF vectors
  4. For a user question, compute cosine similarity with every FAQ
  5. Return the answer of the best match (or a fallback if the match is weak)

Uses NLTK's PorterStemmer when NLTK is installed, otherwise a small
built-in stemmer, so the project also runs without NLTK.
"""

import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:  # Optional: use NLTK if available
    from nltk.stem import PorterStemmer

    _stemmer = PorterStemmer()

    def stem(word: str) -> str:
        return _stemmer.stem(word)

except ImportError:  # Fallback: very small suffix stripper

    def stem(word: str) -> str:
        for suffix in ("ing", "ed", "es", "s", "ly"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)]
        return word


STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "was", "were", "be", "been", "do",
    "does", "did", "i", "me", "my", "we", "you", "your", "it", "its", "of",
    "to", "in", "on", "at", "for", "and", "or", "if", "can", "could", "would",
    "should", "will", "shall", "please", "tell", "about", "this", "that",
    "there", "any", "so", "as", "by", "with", "from", "have", "has", "had",
    "get", "what", "which", "who", "whom", "how", "when", "where", "why",
}

# Words that matter in questions, so we keep them even though they look like stopwords
KEEP = {"when", "where", "how", "what"}

GREETINGS = {"hi", "hello", "hey", "hii", "hola", "namaste", "good morning",
             "good evening", "good afternoon"}
THANKS = {"thanks", "thank you", "thankyou", "thx", "ok thanks", "okay thanks"}
BYE = {"bye", "goodbye", "see you", "exit", "quit"}


def preprocess(text: str) -> str:
    """Lowercase, strip punctuation, tokenize, remove stopwords, stem."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS or t in KEEP]
    tokens = [stem(t) for t in tokens]
    return " ".join(tokens)


class FAQChatbot:
    def __init__(self, faq_path: str | Path | None = None, threshold: float = 0.20):
        if faq_path is None:
            faq_path = Path(__file__).parent / "faqs.json"
        with open(faq_path, encoding="utf-8") as f:
            self.faqs = json.load(f)

        self.threshold = threshold
        self.questions = [item["question"] for item in self.faqs]
        self.answers = [item["answer"] for item in self.faqs]
        self.alternates = [" ".join(item.get("alt", [])) for item in self.faqs]

        # Index on question + alternate phrasings + answer text,
        # so synonyms and keyword hits in answers also help matching
        corpus = [
            preprocess(f"{q} {q} {alt} {a}")
            for q, alt, a in zip(self.questions, self.alternates, self.answers)
        ]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(corpus)

    def get_response(self, user_input: str) -> dict:
        """Return {'answer': str, 'matched_question': str|None, 'score': float}."""
        cleaned_raw = user_input.strip().lower()

        if not cleaned_raw:
            return {"answer": "Please type a question.", "matched_question": None, "score": 0.0}
        if cleaned_raw in GREETINGS:
            return {"answer": "Hello! I am the CodeAlpha Internship FAQ bot. Ask me about tasks, deadlines, submission, certificates and more.",
                    "matched_question": None, "score": 1.0}
        if cleaned_raw in THANKS:
            return {"answer": "You're welcome! All the best for your internship.",
                    "matched_question": None, "score": 1.0}
        if cleaned_raw in BYE:
            return {"answer": "Goodbye! Good luck with your tasks.",
                    "matched_question": None, "score": 1.0}

        query = preprocess(user_input)
        if not query:
            return self._fallback()

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        best = int(scores.argmax())
        best_score = float(scores[best])

        if best_score < self.threshold:
            return self._fallback(best_score)

        return {
            "answer": self.answers[best],
            "matched_question": self.questions[best],
            "score": round(best_score, 3),
        }

    @staticmethod
    def _fallback(score: float = 0.0) -> dict:
        return {
            "answer": ("Sorry, I could not find an answer to that. Try rephrasing, "
                       "or contact CodeAlpha support on WhatsApp: +91 9336576683."),
            "matched_question": None,
            "score": round(score, 3),
        }


def main() -> None:
    bot = FAQChatbot()
    print("CodeAlpha FAQ Chatbot (type 'exit' to quit)\n")
    while True:
        try:
            user = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        result = bot.get_response(user)
        print(f"Bot: {result['answer']}\n")
        if user.strip().lower() in BYE:
            break


if __name__ == "__main__":
    main()
