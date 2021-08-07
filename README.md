## Style transfer

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
