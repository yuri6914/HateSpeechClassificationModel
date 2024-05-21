from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
import googleapiclient.discovery
import pandas as pd

app = Flask(__name__)

# YouTube API 설정
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyChxC7sWyzVO5Rl4zAm5URHS7RligGJ6M8"

# 모델 및 토크나이저 로드
model_path = "data/korean_hate_serch_Model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# YouTube API 클라이언트 생성
youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)

# 댓글 분류 함수
def classify_comments(comments):
    dataset = CommentDataset(comments, tokenizer)
    loader = DataLoader(dataset, batch_size=16)
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
    return predictions

# 댓글 가져오기
def get_comments(video_id):
    comments = []
    comment_authors = []
    comment_likes = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,  # API 제한에 따라 조정 가능
        textFormat="plainText",
        order="relevance"  # 좋아요가 높은 순서로 정렬
    )
    response = request.execute()

    for item in response.get("items", []):
        text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"].replace('\n', ' ')  # \n 제거
        likes = item["snippet"]["topLevelComment"]["snippet"]["likeCount"]
        author = item["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"]
        comments.append(text)
        comment_authors.append(author)
        comment_likes.append(likes)
    return comments, comment_authors, comment_likes

# 데이터셋 정의
class CommentDataset(Dataset):
    def __init__(self, comments, tokenizer, max_len=256):
        self.comments = comments
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        comment = self.comments[idx]
        inputs = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0)
        }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_url = request.form['video_url'].split('&')[0]  # '&' 이후의 내용 제거
        video_id = video_url.split('=')[-1]
        comments, comment_authors, comment_likes = get_comments(video_id)
        predictions = classify_comments(comments)
        
        # 혐오표현이 1인 댓글과 0인 댓글을 나누기
        hate_comments = []
        normal_comments = []
        hate_authors = []
        normal_authors = []
        hate_likes = []
        normal_likes = []
        for comment, author, likes, prediction in zip(comments, comment_authors, comment_likes, predictions):
            if prediction == 1:
                hate_comments.append(comment)
                hate_authors.append(author)
                hate_likes.append(likes)
            else:
                normal_comments.append(comment)
                normal_authors.append(author)
                normal_likes.append(likes)
        
        hate_data = pd.DataFrame({
            '댓글 작성자': hate_authors,
            '댓글 내용': hate_comments,
        })
        
        normal_data = pd.DataFrame({
            '댓글 작성자': normal_authors,
            '댓글 내용': normal_comments,
        })
        
        hate_html = hate_data.to_html(index=False, justify='center')
        normal_html = normal_data.to_html(index=False, justify='center')
        return render_template('result.html', hate_data=hate_html, normal_data=normal_html)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
