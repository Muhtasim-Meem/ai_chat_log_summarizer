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
        self.stop_words = set(stopwords.words('english') + ['can', 'could', 'would', 'should', 'may', 'might', 'must'])



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
        

    def analyze_chat_statistics(self):

        return {
            'user_messages' : len(self.user_messages),
            'ai_messages' : len(self.ai_messages),
            'total_messages' : len(self.user_messages) + len(self.ai_messages),
            'exchanges' : min(len(self.user_messages), len(self.ai_messages)),
        }


    def extract_keywords(self,use_tfidr = True, top_n= 5):

        all_text = ' '.join(self.user_messages + self.ai_messages)

        if not use_tfidr:

            tokens = word_tokenize(all_text.lower())
            tokens = [
                word for word in tokens
                if word not in string.punctuation and word not in self.stop_words
                and len(word) > 2
            
        ]
            
            keywords = Counter(tokens).most_common(top_n)
            methos = "freequency based"   

        else:

            vectorizer = TfidfVectorizer(lowercase = True, stop_words = 'english',min_df = 1)
            corpus = self.user_messages + self.ai_messages
            tfidf_matrix = vectorizer.fit_transform(corpus)
            feature_names = vectorizer.get_feature_names_out()

            word_scores = {}

            for i in range(len(corpus)):
                feature_idx = tfidf_matrix[i, :].nonzero()[1]
                tfidf_scores = zip(feature_idx, [tfidf_matrix[i, x] for x in feature_idx])

                for idx, score in tfidf_scores:
                    word = feature_names[idx]
                    if word not in self.stop_words and len(word) > 2:
                        if word not in word_scores:
                            word_scores[word] = score
                        else:
                            word_scores[word] += score

            keywords = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
            methos = "tf-idf"

        return keywords, methos

    def topics_identify(self,keywords ):
        keyword_str = ' '.join([word for word, _ in keywords])
        topics = {
                'programming': ['python','code', 'program',"develop",'script',"function"],
                'machine learning': ['ai',"ml",'algorithm','learn','model','neural','train'],
                "data science": ['data',"analysis",'statistic','visualization',"pandas",'dataset'],
                'natural language processing': ['nlp', 'language', 'text', 'tokenize', 'sentiment'],
                'computer vision': ['image', 'vision', 'object', 'detect', 'recognition'],
                "web development": ["web", "html", "css", "javascript", "frontend", "backend"],
                "general inquiry": ["explain", "what", "how", "tell", "mean"]
            }
        topic_scores = {topic: sum(1 for kw in kws if kw in keyword_str.lower()) 
                        for topic, kws in topics.items()}
        max_score = max(topic_scores.values(), default=0)
        if max_score == 0:
            return "General conversation"
            
        main_topics = [topic for topic, score in topic_scores.items() if score == max_score]
        return ", ".join(main_topics)

    def generate_summary(self, use_tfidf=True):
            """Generate a summary of the chat log."""
            stats = self.analyze_chat_statistics()
            keywords, method = self.extract_keywords(use_tfidf)
            topic = self.topics_identify(keywords)
            
            return f"""Summary:
    - The conversation had {stats['exchanges']} exchanges ({stats['total_messages']} messages total).
    - {stats['user_messages']} messages were from the user and {stats['ai_messages']} from the AI.
    - The conversation was primarily about {topic}.
    - Most common keywords (using {method}): {', '.join([word for word, _ in keywords])}.
    """

def process_file(file_path, use_tfidf=True):
    """Process a single chat log file."""
    summarizer = ChatLogSummarizer()
    if summarizer.parse_chat_log(file_path):
        summary = summarizer.generate_summary(use_tfidf)
        print(f"\nSummary for {os.path.basename(file_path)}:")
        print(summary)
        return summary
    return None

def process_folder(folder_path, use_tfidf=True):
    """Process all .txt files in a folder."""
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return {}
    
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    if not txt_files:
        print(f"No .txt files found in {folder_path}")
        return {}
    print(f"Processing {len(txt_files)} chat log files...")
    return {file_name: process_file(os.path.join(folder_path, file_name), use_tfidf) 
            for file_name in txt_files if process_file(os.path.join(folder_path, file_name), use_tfidf)}



