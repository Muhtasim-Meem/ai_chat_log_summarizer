import re
import os
import string
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer



nltk.download('punkt')
nltk.download('stopwords')


class ChatLogSummarizer:
    def __init__(self):
        self.user_messages = []
        self.ai_messages = []
        self.stop_words = set(stopwords.words('english')) + ['can', 'could', 'would', 'should', 'may', 'might', 'must']



def parse_chat_log(self, file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            chat_log = file.read()

        user_pattern = re.compile(r'User:(.*?)(?=AI:|$)', re.DOTALL)
        ai_pattern = re.compile(r'AI:(.*?)(?=User:|$)', re.DOTALL)
        self.user_messages = [m.strip() for m in user_pattern.findall(chat_log)]
        self.ai_messages = [m.strip() for m in ai_pattern.findall(chat_log)]        

        return True
    except Exception as e:
        print(f"Error reading file: {e}")
        return False
    
