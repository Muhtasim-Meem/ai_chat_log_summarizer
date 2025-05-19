# AI Chat Log Summarizer 🧠💬

This Python-based tool reads `.txt` chat logs between a user and an AI, parses the conversation, and generates a simple summary with message stats and keyword analysis.

## ✅ Features
- Separates messages by speaker (User/AI)
- Counts total messages and split by speaker
- Extracts top 5 keywords (excluding stopwords)
- Bonus: TF-IDF keyword extraction & multi-file support

## 📁 Sample Chat Format



## 🛠 How to Run
```bash
pip install -r requirements.txt
python main.py chat.txt


Summary:
- 15 exchanges
- User asked about Python and its uses
- Common keywords: Python, use, data, AI, language
