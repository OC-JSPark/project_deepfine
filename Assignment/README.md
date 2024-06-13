## Assignment 1
AI NLP, Vision 모델 각 1개를 선정하여 파인튜닝 후 학습된 모델을 자동으로 배포하여 서비스하는 서버를 만들어보세요.

- Requirements
1. 파인튜닝 후 모델은 평가를 거쳐 가장 높은 점수를 획득한 모델을 배포해야합니다.
2. 파인튜닝, 평가, 배포, 모니터링을 중간의 사람의 개입이 없는 모든 프로세스가 자동화된 파이프라인 형태로 구현해야 합니다.
3. 사용하실 AI 모델의 입출력은 선택하신 모델에 따라 자율적으로 선택하셔서 구현하시면 됩니다.
4. NLP, Vision 태스크는 서로 독립적으로 구현하시면 됩니다.
5. 파인튜닝 후 모델 성능이 반드시 높을 필요는 없습니다.

-> Assignment1 / vision
-> Assignment1 / nlp
    * issue) local pc docker hub error로인한 최종 test 미확인.
    
## Assignment 2
Transformer 모델의 Scaled dot-product 어텐션 메커니즘을 구현해보세요. 그리고 ViT(Vision Transformer)에서 이 부분이 어떤 의미로 동작하는지 임의의 숫자를 넣어 만든 Tensor 데이터를 활용하여 간단한 코드를 작성하여 설명하세요.

- Requirements
1. C++ 또는 Python 이용하여 작성하세요.
2. TensorFlow, Pytorch, JAX 등 딥러닝 프레임워크 사용 가능 (Keras와 같은 상대적으로 고수준의 API는 사용 불가)
3. 그 외에 라이브러리 사용 불가능

-> deepfine_project_Assignment2.ipynb

## Assignment 3
임의의 이미지 데이터를 이용하여 ViT 모델의 전처리 과정이 CNN 모델과 어떻게 다른지 코드를 통해 비교하여 구현하세요.

- Requirements
1. C++ 또는 Python 이용하여 작성하세요.
2. TensorFlow, Pytorch, JAX 등 딥러닝 프레임워크 사용 가능 (Keras와 같은 상대적으로 고수준의 API는 사용 불가)
3. 이미지 로딩 및 임베딩을 얻기 위한 라이브러리 외에 사용 불가능

-> deepfine_project_Assignment3.ipynb
