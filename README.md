# PoseNet Pytorch 구현 및 ROS 연동

---

## 사용코드

- https://github.com/youngguncho/PoseNet-Pytorch

---

## 필요항목 설치

- Python 3.7버전 이상으로 사용

- Python 3.9를 사용했는데, pip3가 깨지는 오류가 있다고 한다.

```bash
apt install python3.9-distutils
pip3 install --upgrade setuptools
pip3 install --upgrade pip
pip3 install --upgrade distlib
```

- Pytorch
```bash
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```

- Pandas
```bash
pip3 install pandas
```

- Numpy
```bash
pip3 install numpy
```

- Matplotlib
```bash
pip3 install matplotlib
```

- Tensorflow
```bash
# CPU만 있는 컴퓨터이기 때문에, -cpu 사용
pip install tensorflow-cpu
```

- Tensorboard
```bash
pip3 install tensorboard
pip3 install tensorboardx
```

- 본인은 CPU뿐이기에 이렇게 했지만, 만약에 GPU와 CUDA를 사용한다면, 홈페이지 참조
- https://pytorch.org/

---

## 사용 과정

- Git에서 코드를 Clone하고, Data Set을 다운로드받음
```bash
wget https://www.repository.cam.ac.uk/bitstream/handle/1810/251342/KingsCollege.zip?sequence=4&isAllowed=y
```

### Training

- Dataset을 ./posenet으로 이동 후 압축해제

- model training을 진행함

```bash
python3 train.py --image_path ./posenet/KingsCollege --metadata_path ./posenet/KingsCollege/dataset_train.txt
```

- encoding 에러가 나서 `data_loader.py`의 24line에 `encoding="latin1"`을 추가해줌

- `model.py`의 `base_model = models.resnet34(pretrained=True)`를 `weights='ResNet34_Weights.DEFAULT'`로 수정함

- `solver.py`의 138~142 line 을 198 line의 밑으로 이동시킴

# tensorboard 실행
tensorboard --logdir ./summaries_{Last Folder name of image path}
```