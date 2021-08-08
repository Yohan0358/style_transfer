## Style transfer

![](https://github.com/Yohan0358/style_transfer/blob/master/output/style1_output.png?raw=true)
![](https://github.com/Yohan0358/style_transfer/blob/master/output/style2_output.png?raw=true)
![](https://github.com/Yohan0358/style_transfer/blob/master/output/style3_output.png?raw=true)

[관련 논문 : Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbh4iLj%2FbtqWXbw5skj%2FsO3IeXJiIa92z9rwKA6M40%2Fimg.png)

- Neral Style Transfer : style image와 content image가 있을 때, Content image는 유지하면서 style만 기존의 imag에 입히는 것
- 유명 화가의 화풍만 가져와 새로운 그림을 그리는 방식이 style transfer임

![](https://www.popit.kr/wp-content/uploads/2018/04/gatys-feature-extraction-1024x711.png)
- pretrained model의 Conv layer에서 feature map을 추출하여 style transfer를 적용함
- style image의 feature map : layer가 깊어질 수록 feature map은 스타일(화풍)에 대한 것
- Content image의 feature map : layer가 깊어질 수록 featuer map은 형태(구조)에 대한 것

- style image와 content image의 feature map을 잘 섞으면 style이 transfer된 image를 생성할 수 있음

## Loss function
![](https://www.popit.kr/wp-content/uploads/2018/04/gatys_algoritm_paper-1024x583.png)
- style image(a), content image(p), 생성될 image(x)
- Content loss
  - pretrained model(여기서는 VGG16)의 conv layer에서 feature map을 추출
  - 동일하게 생성될 x에서도 feature map 추출
  - p의 feature map(P)와 x의 feature map(F)의 거리를 Loss function으로 정의
 
<img src="https://www.popit.kr/wp-content/uploads/2018/05/gatys_content_loss.png" width="30%" height="20%"></img>

- Style loss
  - Content loss와 마찬가지로 a와 x에 대해서 feature map 추출
  - 각 feature map의 gram matrix 구함
  * gram matirx : 채널간의 상관성(correlation)을 표현한 matrix
  * style image와 생성될 image가 동일한 상관성을 가지도록 학습
  
 <img src="https://www.popit.kr/wp-content/uploads/2018/05/gatys_style_loss_total.png" width="60%" height="50%"></img>
 
 - Total loss
 - Content loss와 style loss에 대한 가중치 적용(여기서는 1/10^4 적용함)
 - Loss funtion의 backward를 통해서 x를 업데이트 시킴

![](https://www.popit.kr/wp-content/uploads/2018/05/gatys_total_loss.png)
