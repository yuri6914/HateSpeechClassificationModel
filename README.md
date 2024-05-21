# HateSpeechClassificationModel
 KGITBANK 데이터 분석 및 챗봇 AI 수업 4차 팀 프로젝트 모델 및 코드

팀원 : 공대윤(본인), 서윤성, 임세동, 노형진

https://kli.korean.go.kr/benchmark/taskOrdtm/taskList.do?taskOrdtmId=56&clCd=ING_TASK&subMenuId=sub01


  국립국어원에서 출제한 과제 중 하나인 혐오발언 탐지를 조별 과제 주제로 선정하여, 그 곳에서 제공하는 데이터를 토대로
 혐오발언을 감지, 탐지하고 분류하는 이진법 딥러닝 모델을 만들고자 하였습니다.

  최초의 목표는 적절한 모델을 선택하여, 제공된 데이터를 정제하고 전처리하여, 혐오 표현을 감지하고 분류하며, 사용된 문맥에 따라
 유동적으로 더 정확하게 혐오표현인지 아닌지 분류 가능한 모델을 구축하는 것 입니다.

  그 이유는, 같은 비속어 및 난폭한 표현이어도 문맥에 따라 그것이 혐오 표현으로 사용될수도있고, 아닐수도 있다고 판단하였기 때문입니다.

 BERT, KoBERT, KoELECTRA, CNN 외 여러가지 모델을 사용했었지만, 눈에 보이는 수치상으로는 val_acc(정확도)가 85프로 이하로 실제 결과물을 떠나서 순위에서도 많이 밀리는 낮은 수치를 기록해 지속적으로 여러 모델을 거쳐가며 확인해보았습니다. 이후, KcBERT및 KcELECTRA 그리고 KoELECTRA_v3 버전을 사용해 각각 92가 넘는 정확도를 보여 세가지 모델을 파인튜닝, 및 하이퍼 파라미터 조절을 통해 조정하기로 결정하였습니다.
  
  이후 목적은 Valid Accuracy를 95퍼 이상으로 끌어올려, 과제 제출 인원 순위 중 상위권을 차지하는 것이 목표였습니다. 지속적인 수치 조정 후, 95프로가 넘는 정확도를 보여 그것을 과제로서 국립국어원에 정식 과제의 결과물로서 제출하고, 평가 점수 96.4285714로 2024.04.25일부터 순위권 2위를 유지하고 있습니다.

(순위권 링크)
 -팀 이름: KGIT_AI팀
 
https://kli.korean.go.kr/benchmark/taskOrdtm/taskLeaderBoard.do?taskOrdtmId=56&clCd=ING_TASK&subMenuId=sub04

 최종적으로 제공 받은 데이터를 증강하고 정제하여 더 좋은 결과물을 만들며, 실제 결과물을 관측해 필요한 조정을 지속적으로 할 예정입니다.

 이후, 저희는 API를 이용해 구글 크롬 확장 프로그램으로서 만들고, 인터넷으로 유투브 및 SNS 이용시 혐오 발언이 탐지되는 글들을 실시간으로 탐지해 검열하여 사용자의 시선에 보이지 않게하는 작업을 연구 중입니다.

  하지만 그 작업은 현재 지속적으로 연구 중이며, 당장에는 유투브 URL을 입력시 URL내의 영상의 댓글들을 가져와서 탐지하고 혐오 표현이 들어있는 댓글인지 아닌지 0과 1로 분류해주는 웹으로 제작해두었습니다.



![KakaoTalk_20240521_161305044_01](https://github.com/yuri6914/HateSpeechClassificationModel/assets/66938653/68db6bc0-b0b5-475a-ac6b-c6d0b25c3dff)
![KakaoTalk_20240521_161305044](https://github.com/yuri6914/HateSpeechClassificationModel/assets/66938653/b745d15c-86a9-43fe-a780-9cf96a0ced49)

