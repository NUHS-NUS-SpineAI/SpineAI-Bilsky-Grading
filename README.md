# SpineAI-Bilsky-Grading
<h1 align="center">
  <p align="center">SpineAI Paper with Code</p>
  <img src="imgs/spineAI-logo.png" alt="SpineAI-logo" height="150">
</h1>

## Dataset
Coming soon!

## Bilsky-Grading model
An overview of the proposed Bilsky-Grading model.
<div align=center><img height="400" src="imgs/model.png"></div>

## Environment

- Python==3.9
- Pytorch==1.9.1
- Keras==2.2.2

## Run the code
bash train.sh

## Training visualization

$ tensorboard --logdir path_to_current_dir/logs

## Training visualization
Internal Test Set
| Normal | Abnormal | Avg Acc |
| ----- | ------ | ------ | 
| 93.58 | 97.62 | 95.6 |


External Test Set
| Normal | Abnormal | Avg Acc |
| ----- | ------ | ------ | 
| 98.12 | 89.94 | 94.03|


## 🤝 Referencing and Citing SpineAI

If you find our work useful in your research and would like to cite our Radiology paper, please use the following citation:

```
@article{hallinan2021deep,
  title={Deep learning model for automated detection and classification of central canal, lateral recess, and neural foraminal stenosis at lumbar spine MRI},
  author={Hallinan, James Thomas Patrick Decourcy and Zhu, Lei and Yang, Kaiyuan and Makmur, Andrew and Algazwi, Diyaa Abdul Rauf and Thian, Yee Liang and Lau, Samuel and Choo, Yun Song and Eide, Sterling Ellis and Yap, Qai Ven and others},
  journal={Radiology},
  pages={204289},
  year={2021},
  publisher={Radiological Society of North America}
}
```

## :mailbox: Contact

Address correspondence to J.T.P.D.H. (e-mail: james_hallinan AT nuhs.edu.sg)

### _Disclaimer_

_This code base is for research purposes and no warranty is provided. We are not responsible for any medical usage of our code._


