# Robô Unitree Go2 EDU

Esse projeto utiliza o robô desenvolvido pela empresa Unitree Robotics, no modelo Go2 EDU, que é um robô quadrúpede projetado para diversas aplicações, incluindo pesquisa, inspeção e entretenimento. O robô é equipado com sensores avançados e câmeras que permitem a navegação autônoma e a interação com o ambiente.

## Interface de Controle

A interface de controle escolhida para operar o robô é o [Unitree Go2 ROS2 SDK](https://github.com/abizovnuralem/go2_ros2_sdk), que consiste em um projeto open-source desenvolvido em Python. Essa interface permite o controle do robô através do framework ROS2 (Robot Operating System 2), facilitando a comunicação entre o software de controle e o hardware do robô.

Devido à necessidade do projeto de locomoção do robô, toda a comunicação com o robô foi realizada utilizando o protocolo WebRTC(Wi-Fi), que oferece baixa latência e alta eficiência na transmissão de dados em tempo real. Sendo assim, o robô pode ser controlado remotamente com precisão e rapidez.

## Problemas Encontrados

Durante o desenvolvimento do projeto, houveram diversos problemas ocasionados principalmente devido a políticas de segurança da universidade em que o projeto foi desenvolvido. Essas políticas restringiram o acesso do robô a rede Wi-Fi da instituição, tendo em vista que o registro na rede é realizado através do aplicativo proprietário da empresa Unitree Robotics, o que impossibilitou a conexão do robô à Internet.

Como alternativa, foi utilizado um dispositivo móvel (celular) para criar um ponto de acesso Wi-Fi, permitindo a conexão do robô à Internet. No entanto, essa solução apresentou limitações de alcance e estabilidade da conexão, o que impactou negativamente na performance do controle remoto do robô.
