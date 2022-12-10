import requests
import json

# 메시지를 보내는 부분. 함수 안 argument 순서 :
# token : Slack Bot의 토큰
# channel : 메시지를 보낼 채널 #stock_notice
# text : Slack Bot 이 보낼 텍스트 메시지. 마크다운 형식이 지원된다.
# attachments : 첨부파일. 텍스트 이외에 이미지등을 첨부할 수 있다.

def notice_message(token, channel, text, attachments):
    attachments = json.dumps(attachments) # 리스트는 Json 으로 덤핑 시켜야 Slack한테 제대로 간다.
    response = requests.post("https://slack.com/api/chat.postMessage",
        headers={"Authorization": "Bearer "+token},
        data={"channel": channel, "text": text ,"attachments": attachments})

a='12:00:12'
Token = 'xoxb-4494660617701-4500069866596-jbojYdi0lJojewSG73xzgsLv' # 자신의 Token 입력
str1_title = 'Job information'
str1_contents=f"The Job is Finished\n Time is {a}"
str2 = "The Job is Finshed \n Cluster is Aurora"

attach_dict = {
    'color' : '#ff0000',
    'author_name' : 'Slack Bot Notice',
    'title' : str1_title,
    'text' : str1_contents,
} # attachment 에 넣고싶은 목록들을 딕셔너리 형태로 입력

attach_list=[attach_dict] # 딕셔너리 형태를 리스트로 변환
notice_message(Token, "#notice-job", str2, attach_list)