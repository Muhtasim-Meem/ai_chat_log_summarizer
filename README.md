# AI Chat Log Summarizer 

This Python-based tool reads `.txt` chat logs between a user and an AI, parses the conversation, and generates a simple summary with message stats and keyword analysis.

## ✅ Features
- Separates messages by speaker (User/AI)
- Store messages in appropriate structures for further analysis.

- Counts total messages and split by speaker
- Exclude common stop words (e.g., "the", "is", "and").
- Extracts top 5 keywords (excluding stopwords)
- Generate Summary (Total number of Exchange, Nature of the conversation,Most common keywords)
- Bonus: TF-IDF keyword extraction & multi-file support

## 📁 Sample Chat Format
- User: Hello!
- AI: Hi! How can I assist you today?
- User: Can you explain what machine learning is?
- AI: Certainly! Machine learning is a field of AI that    allows systems to learn from data.



##  How to Run
1. Clone the Repository
```bash
git clone https://github.com/Muhtasim-Meem/ai_chat_log_summarizer.git
cd ai_chat_log_summarizer
```
2. Create and Activate Virtual Environment (optional but recommended)

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```
3. Install Dependencies
```bash
pip install -r requirements.txt
```
4. Download NLTK Resources

Run this Python code once:
- import nltk
- nltk.download('punkt')
- nltk.download('stopwords')

5.  Run the Script For a single chat log file:
```bash
python ai_chat_log_summarizer.py chat.txt
```
To use basic keyword extraction instead of TF-IDF:
```bash
python ai_chat_log_summarizer.py chat.txt --basic
```
To process multiple chat logs in a folder:
```bash
python ai_chat_log_summarizer.py path/to/folder/
```


- Summary for chat.txt:
- Summary:
-    The Conversation had 2 exchanges (4 messages total).
-    2 messages were from the user and 2 from the AI.
-    The Conversation was primarily about machine learning.
-    Most Common keywords (using freequency based): machine, learning, hello, explain, assist.
