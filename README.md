# singular-points


# Downloads
- !wget https://github.com/kornia/data/raw/main/simba.png
- !wget https://github.com/kornia/data/raw/main/arturito.jpg

## Install
- pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
- pip3 install kornia e2cnn matplotlib scikit-image 


# arquivos

- antigo : sem o kornia
  - file model: 'model_antigo.pt'
- train : esta executando com pessima performance
  - file model: 'model_train.pt'

# Loss
- A função de perda SSIM ela agrupa informação de contraste , iluminação e estrutura, se aproximando mais da percepção humana.
  - a margim utilizada na função tripletloss é 1
- A função de perda MSE faz uma a diferença quadratica fazendo uma análise pixel a pixel.
  - a margim utilizada na função tripletloss é 2
- No calculo de perda a mesma imagem é computada de diferentes pespectivas e comparada se o mapa de ativação é o mesmo considerendo a projeção.
- A função de perda deve maximizar a similaridade quando a imagem é a mesma e minimizar quando for distinta.
- A mascara garante que só iremos observar as regioes visiveis da imagem.
- A borda da imagem concentra uma região de mudança brusca de gradiente ao remover essa região do mapa de ativação, excluimos uma região de atenção indesejada. 

# Polling
- O polling auxilia na diminuição do tempo do treinamento pois reduz a dimensionalidade, no entanto temos que usar com cuidado quando queremos trabalhar com identificação.
- O max polling parece ser mais util pois realça as regioes de maximos da imagem
- O avg minimizava o ruído mas apaga algumas regiões do mapa de ativação.

