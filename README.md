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

```bash
#tensorboard 실행
tensorboard --logdir ./summaries_{Last Folder name of image path}
```

- 이후, model test를 진행함

```bash
python3 test.py --image_path ./posenet/KingsCollege --metadata_path ./posenet/KingsCollege/dataset_test.txt --test_model 49

python3 test.py --image_path ./posenet/KingsCollege --metadata_path ./posenet/KingsCollege/dataset_test.txt --test_model 399

python3 test.py --image_path ./posenet/KingsCollege --metadata_path ./posenet/KingsCollege/dataset_test.txt --test_model best
```


---

## 결과

- test.py를 돌려본 결과값은 x,y,z좌표값과 Quaternion좌표값을 ZYX-Euler Angle 값으로 변환한 값을 보여주었다.

- 오차평균
  - `49_net.pth`: 1.671 / 0.121
  - `399_net.pth`: 22.280 / 0.528
  - `best_net.pth`: 2.161 / 0.106

- 학습을 더 오래 진행한 결과값이 더 오차가 크지만, Val loss와 train loss의 차이가 가장 적다.

- 후에 ROS에 적용 할때는 사진을 `sensor_msgs/Image.msg`로 Subscribe하고, Quaternion 좌표값을 ZYX-Euler Angle 값으로 변환하지 않고, `geometry_msgs/Pose.msg`의 `Point position`에 x,y,z좌표값을 `Quaternion orientation`에 Quaternion 좌표값을 입력하여 Publish 하는 Node를 제작하면 되겠다고 생각하였다.