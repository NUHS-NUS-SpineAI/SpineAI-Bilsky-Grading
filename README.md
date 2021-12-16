# SpineAI-Bilsky-Grading
<h1 align="center">
  <p align="center">SpineAI Paper with Code</p>
  <img src="imgs/spineAI-logo.png" alt="SpineAI-logo" height="150">
</h1>


## Bilsky-Grading model

<div align=center><img height="400" src="imgs/model.png"></div>

## Environment

- Python==3.9
- Pytorch==1.9.1
- Keras==2.2.2

## Run the code
bash train.sh

## Training visualization

```bash
$ tensorboard --logdir path_to_current_dir/logs

## Results
Internal Test Set
| Normal | Abnormal | Avg Acc |
| ----- | ------ | ------ | 
| 93.58 | 97.62 | 95.6 |
External Test Set

External Test Set
| Normal | Abnormal | Avg Acc |
| ----- | ------ | ------ | 
| 98.12 | 89.94 | 94.03|


## :mailbox: Contact

Address correspondence to J.T.P.D.H. (e-mail: james_hallinan AT nuhs.edu.sg)

### _Disclaimer_

_This code base is for research purposes and no warranty is provided. We are not responsible for any medical usage of our code._


