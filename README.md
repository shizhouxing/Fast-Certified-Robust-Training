# Fast Certified Robust Training with Short Warmup

Interval bound based certified robust training such as [IBP](https://github.com/deepmind/interval-bound-propagation) and [CROWN-IBP](https://github.com/huanzhang12/CROWN-IBP) are one of the most effective approaches for L_inf norm certified robustness. Unfortunately, IBP based training is often unstable and requires a large number of "warmup" epochs and a long training schedule. Existing works typically *require a large number epochs* (e.g., a few thousands) to achieve SOTA certified error. 

We find that weight initialization adopted prior works, which were originally designed for regular network training, can cause exploding certified bounds and are thus not suitable for IBP.  We also find that IBP leads to imbalanced ReLU activation states with the model perfers inactive (dead) ReLU neurons significantly more.

To address the above two issues, we propose the following improvements for certified training, and thereby significantly reduce the number of training epochs while outperforming literatures SOTA:

- We derive a new weight initialization for IBP-based certified training, namely *IBP initialization*, to stabilize certified bounds at initialization. 

- We find that Batch Normalization is a *crucial architectural component* for IBP training, as it normalizes pre-activation values, and thereby can balance ReLU activation states and stabilize variance.

- We enhance the warmup with regularizers to further stabilize the certified bounds and meanwhile balance the ReLU activation states explicitly.

With our proposed **initialization**, **architectural changes**, and **regularizers** combined, we achieved **65.03%** verified error on CIFAR-10 (eps=8/255), **82.13%** verified error on TinyImageNet, and **10.98%** verified error on MNIST (eps=0.4), which noticeably outperforms literature IBP and CROWN-IBP results. Additionally, we need much fewer training epochs to achieve these results: **160 epochs** for CIFAR-10 and **80 epochs** for TinyImageNet. More details can be found in our paper:

[Fast Certified Robust Training with Short Warmup](https://arxiv.org/abs/2103.17268), by Zhouxing Shi\*, Yihan Wang\*, Huan Zhang, Jinfeng Yi and Cho-Jui Hsieh (\* Equal contribution), to appear in *NeurIPS 2021*.

## Dependencies

We use [auto_LiRPA](https://github.com/KaidiXu/auto_LiRPA) as a base framework for robustness verification. It can be installed via `pip install auto-LiRPA==0.2` or source code:

```bash
git clone https://github.com/KaidiXu/auto_LiRPA.git
cd auto_LiRPA
python setup.py install
```

And install other required dependencies:

```bash
pip install -r requirements.txt
```

We are using PyTorch 1.8.1, but we expect the code to work well for other recent versions (since PyTorch 1.6.0).

## How to run

We show sample usages for the three datasets below:

### MNIST

```
python train.py --method={method} --dir=model_mnist --scheduler_opts=start=1,length=20 --lr-decay-milestones=50,60 --num-epochs=70 --config=config/mnist.json --model={model} 
```

where `{method}` can be chosen from `["vanilla", "fast"]` (for Vanilla IBP and our fast training (initialization + regularizers) respectively), and `{model}` can be chosen from `["cnn", "wide_resnet_8", "resnext"]`. 

### CIFAR-10

```
python train.py --method={method} --dir=model_cifar --scheduler_opts=start=2,length=80 --lr-decay-milestones=120,140 --num-epochs=160 --config=config/cifar.json --model={model} 
```

where `{model}` can be chosen from `["cnn", "wide_resnet_8", "resnext"]`. Others are similar to MNIST.

### Tiny-ImageNet

Please prepare the Tiny-ImageNet dataset first with:
```
cd data/tinyimagenet
bash tinyimagenet_download.sh
```

Then
```
python train.py --method={method} --config=config/tinyimagenet.ibp.json --model={model} --scheduler_name=SmoothedScheduler --scheduler_opts=start=2,length=20 --reg-lambda=2e-1 --num-epochs=80  --lr-decay-milestones=61,71 --num-class 200 --grad-acc-steps=2 --batch-size=128
```
where `{model}` can be chosen from `["cnn_7layer_bn_imagenet", "wide_resnet_imagenet64", "ResNeXt_imagenet64"]`.  It is preferred to use a smaller `lambda` for TinyImagenet, by setting `--reg-lambda` (e.g., 0.1, in contrast to the default 0.5 value).

### GPU Memory for Large Models

Since some models are relatively large, when the model cannot fit into the available GPU memory, you may add a `--grad-acc-steps` argument for gradient accmulation. Multi-GPU is not supported so far.

## Checkpoint

We have released some of the [checkpoints](https://drive.google.com/drive/folders/1lWK0JDhsqtCD2FtvEVPstcFCUo_mlGCg?usp=sharing) of our training.
