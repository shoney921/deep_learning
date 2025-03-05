# 『밑바닥부터 시작하는 딥러닝 1(리마스터판)』

<a href="https://www.hanbit.co.kr/channel/category/category_view.html?cms_code=CMS2416204088&cate_cd="><img src="https://github.com/WegraLee/deep-learning-from-scratch/blob/master/remastered.png" width="300" align=right></a>

<a href="https://product.kyobobook.co.kr/book/preview/S000215599933">[미리보기]</a> | <a href="https://docs.google.com/document/d/1kOK8Tu4E0ENtl0N5ugSVH8B7kfIJ53Uiarx52T5RIqE/">[알려진 오류(정오표)]</a> | <a href="https://github.com/WegraLee/deep-learning-from-scratch/raw/refs/heads/master/equations_and_figures.zip">[본문 그림과 수식 이미지 모음]</a>

:red_circle: **[공지]** 2025년 1월에 '리마스터판'이 출간되었습니다. 리마스터판의 예제 코드는 초판의 예제 코드와 호환되므로, 초판 독자 분들도 이 저장소의 코드를 그대로 활용하시면 됩니다. 또한, 본질이 달라진 게 아니므로 다시 구매하실 필요 없습니다. 달라진 점은 <a href="https://www.hanbit.co.kr/channel/category/category_view.html?cms_code=CMS2416204088&cate_cd=">제작 뒷이야기</a>에 정리했습니다.

:red_circle: **[공지]** 종종 실습용 손글씨 데이터셋 다운로드 사이트( http://yann.lecun.com/exdb/mnist/ )가 연결되지 않습니다.
그래서 예제 수행에 필요한 데이터셋 파일을 /dataset/ 디렉터리에 올려뒀습니다.
혹 사이트가 다운되어 데이터를 받을 수 없다면 아래 파일 4개를 각자의 <예제 소스 홈>/dataset/ 디렉터리 밑에 복사해두면 됩니다.

* [t10k-images-idx3-ubyte.gz](https://github.com/WegraLee/deep-learning-from-scratch/raw/master/dataset/t10k-images-idx3-ubyte.gz)
* [t10k-labels-idx1-ubyte.gz](https://github.com/WegraLee/deep-learning-from-scratch/raw/master/dataset/t10k-labels-idx1-ubyte.gz)
* [train-images-idx3-ubyte.gz](https://github.com/WegraLee/deep-learning-from-scratch/raw/master/dataset/train-images-idx3-ubyte.gz)
* [train-labels-idx1-ubyte.gz](https://github.com/WegraLee/deep-learning-from-scratch/raw/master/dataset/train-labels-idx1-ubyte.gz)

---

## 파일 구성
|폴더 이름 |설명                         |
|:--        |:--                          |
|ch01       |1장에서 사용하는 소스 코드 |
|ch02       |2장에서 사용하는 소스 코드    |
|...        |...                          |
|ch08       |8장에서 사용하는 소스 코드    |
|common     |공통으로 사용하는 소스 코드  |
|dataset    |데이터셋용 소스 코드 |

## 주피터 노트북
이 책의 코드는 주피터 노트북으로도 제공됩니다. 다음 링크를 클릭하면 구글 콜랩에서 노트북을 실행할 수 있습니다.
| 장 | 구글 콜랩에서 열기 |
| :--- | :--- |
| 1장 헬로 파이썬 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1p_pBspaD0U2lDddKQj3m0HS0gIcJ4GNH?usp=drive_link) |
| 2장 퍼셉트론 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AUKNoYENHXAsnA6lQikOAm_Hx6Tc2zTk?usp=drive_link) |
| 3장 신경망 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1k-R8IvCp2rdMdYMuZ84HcoM0lCzxCfPP?usp=drive_link) |
| 4장 신경망 학습 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yftb83a6nGRuxNUOwnBjPMAV1h7BM8w8?usp=drive_link) |
| 5장 오차역전파법 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ocP0omkB2QY9yg1KOqgFtq9R8ApY-7RV?usp=drive_link) |
| 6장 학습 관련 기술들 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cj88ux7BKceGK3vt-VB5dHu8oBPq2FDQ?usp=drive_link) |
| 7장 합성곱 신경망(CNN) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SQyMbVbqNvmODy4_CIFlcFfgUH9R2oIt?usp=drive_link) |
| 8장 딥러닝 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GXNAB0hnlbBGtsmS0BnSmTavQLAYyz7d?usp=drive_link) |


## 요구사항
소스 코드를 실행하려면 아래의 소프트웨어가 설치되어 있어야 합니다.

* 파이썬 3.x
* NumPy
* Matplotlib


## 실행 방법 (리마스터판)
어디서든 실행할 수 있습니다.
```
$ python deep-learning-from-scratch-master/ch01/img_show.py

$ cd deep-learning-from-scratch-master
$ python ch01/img_show.py

$ cd ch01
$ python img_show.py
```

## 실행 방법 (초판)
각 장의 디렉터리로 이동 후 실행해야 합니다(**다른 디렉터리에서는 제대로 실행되지 않을 수 있습니다!**).
```
$ cd ch01
$ python man.py

$ cd ../ch05
$ python train_nueralnet.py
```

---

## 동영상 강의
수원대학교 한경훈 교수님께서 『밑바닥부터 시작하는 딥러닝』 1, 2편을 교재로 진행하신 강의를 공개해주셨습니다. 책만으로 부족하셨던 분들께 많은 도움이 되길 바랍니다.

딥러닝 I - <a href="https://sites.google.com/site/kyunghoonhan/deep-learning-i">[강의 홈페이지]</a>

[![시리즈 1](https://img.youtube.com/vi/8Gpa_pdHrPE/0.jpg)](https://www.youtube.com/watch?v=8Gpa_pdHrPE&list=PLBiQZMT3oSxW1RS1hn2jWBgswh0nlcgQZ)

딥러닝 II - <a href="https://sites.google.com/site/kyunghoonhan/deep-learning-ii">[강의 홈페이지]</a>

[![시리즈 1](https://img.youtube.com/vi/5fwD1p9ymx8/0.jpg)](https://www.youtube.com/watch?v=5fwD1p9ymx8&list=PLBiQZMT3oSxXNGcmAwI7vzh2LzwcwJpxU)

딥러닝 III - <a href="https://sites.google.com/site/kyunghoonhan/deep-learning-iii">[강의 홈페이지]</a>

[![시리즈 1](https://img.youtube.com/vi/kIobK76on3s/0.jpg)](https://www.youtube.com/watch?v=kIobK76on3s&list=PLBiQZMT3oSxV3RxoFgNcUNV4R7AlvUMDx)

---

## 팬픽 - 바닷속 딥러닝 어드벤처 (5부작)

<img src="https://github.com/WegraLee/deep-learning-from-scratch-5/blob/main/posters/%E1%84%87%E1%85%A1%E1%84%83%E1%85%A1%E1%86%BA%E1%84%89%E1%85%A9%E1%86%A8%20%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%8B%E1%85%A5%E1%84%83%E1%85%B3%E1%84%87%E1%85%A6%E1%86%AB%E1%84%8E%E1%85%A5.png?raw=true">

"<밑바닥부터 시작하는 딥러닝>의 주인공 생선들은 딥러닝 기술로 바닷속 생태계를 어떻게 혁신하고 있을까요? 어공지능의 첨단을 이끌어가는 밑시딥 생선들과 신나는 모험을 떠나보세요."

바닷속 세계를 배경으로, 해양 생물들이 자신의 특성과 필요에 맞는 딥러닝 기술을 개발하여 문제를 해결해 나가는 모험을 그린 연작 소설입니다. 시리즈를 읽으신 분은 더 많은 재미를 느끼실 수 있도록 딥러닝 요소들을 곳곳에 삽입하였습니다.

각 편의 주인공과 주제는 다음과 같습니다.

1. **시야를 찾아서**: 쏨뱅이(쏨)가 **이미지 처리 기술**을 개발하여 주변 환경을 선명하게 파악
1. **상어공주**: 괭이상어 공주(꽹)가 **자연어 처리** 기술로 돌고래 왕자와의 사랑을 쟁취
1. **DeZero의 창조자**: 나뭇잎해룡(잎룡)이 **딥러닝 프레임워크**를 만들어 기술 보급과 협업 촉진
1. **제발, 가즈아!**: 가자미(가즈아)가 **심층 강화 학습**으로 먹이가 풍부한 새로운 바다 개척
1. **피쉬카소와 천재의 초상**: 유령실고기(피쉬카소)가 **이미지 생성 모델**로 바닷속 예술계 혁신

<a href="https://www.hanbit.co.kr/channel/series/series_detail_list.html?hcs_idx=34">소설 보러 가기</a>

---

## 라이선스

이 저장소의 소스 코드는 [MIT 라이선스](http://www.opensource.org/licenses/MIT)를 따릅니다.
상업적 목적으로도 자유롭게 이용하실 수 있습니다.
