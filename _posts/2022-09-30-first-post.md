---
layout: single
title:  "Github 블로그 만들기"
---

## 깃허브 블로그 만들기-테마 선택

저는 블로그를 만들기위해 구글링 및 유튜브를 찾아보던 중 가장 쉬운 방법인 누군가 올려놓은 테마를 Fork하는 방식을 선택했습니다.  

**Fork**는 다른 사람의 Github repository에서 내가 어떤 부분을 수정하거나 추가 기능을 넣고 싶을 때 해당 respository를 내 Github repository로 그대로 복제하는 기능입니다.  

테마들은 아래 링크에서 고를 수 있었습니다.  
<https://github.com/topics/jekyll-theme>  


위 링크에서 마음에 드는 테마를 Fork 해옵니다.  

<img width="614" alt="스크린샷 2022-10-29 오후 3 55 42" src="https://user-images.githubusercontent.com/113446739/198818442-1d02e462-9136-42bc-b33f-83d73122b2cd.png">  

테마에 대한 더 자세한 정보를 알고싶다면 테마의 메인에 있는 **read.me** 파일을 읽어보면 다양한 기능과 상세한 정보들이 나와있습니다.  

Fork를 진행한 이후 Setting의 General 탭에서 Repository name을 본인의 repository 주소로 꼭 바꿔줘야합니다.

<img width="1196" alt="스크린샷 2022-10-29 오후 5 20 19" src="https://user-images.githubusercontent.com/113446739/198821500-6e2d8318-8f5c-4219-bfde-5fc85813cc56.png">



## 개인의 repository 복제

저는 Visual studio code를 이용해 Github와 연동했습니다.  

1. VS CODE를 열고, 좌측 Clone Repository(리포지토리 복제)를 클릭합니다.  
2. 개인의 Github 주소를 입력하고 Clone form Github를 클릭해 복제합니다.
    
    <img width="1164" alt="스크린샷 2022-10-29 오후 4 31 01" src="https://user-images.githubusercontent.com/113446739/198819749-3ed37143-9254-4e53-aef1-38b23b22387f.png">

3. 이후 화면에서 시키는대로 진행하면 하위 폴더가 생기면서 모든 내용이 복제됩니다.  
4. 연동이 완료된 후 원하는 파일을 수정해 커밋 후 푸시해 파일을 수정하거나 추가합나다.  
    
    <img width="233" alt="스크린샷 2022-10-29 오후 5 09 10" src="https://user-images.githubusercontent.com/113446739/198821132-da58bcfe-0678-4e17-8112-ec52802a4ea8.png">



## 블로그 환경설정

_config.yml 파일을 수정해 블로그의 제목, 배너, 개인 SNS 링크등 다양한 기능을 연동하고 수정할 수 있습니다.  

<img width="481" alt="스크린샷 2022-10-29 오후 5 15 36" src="https://user-images.githubusercontent.com/113446739/198821308-ef2b97b6-5965-4ec1-aa4e-5518484a1884.png">

테마마다 파일 수정방법이 조금씩 다르니 Readme파일에서 찾아서 수정하고 싶은 부분을 수정하면 됩니다.  
위 방법대로 한다면 자신의 블로그가 어느정도 완성된 상태를 확인하실 수 있을 것입니다.
