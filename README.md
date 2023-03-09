# DICTION:	DynamIC robusT whIte bOx watermarkiNg scheme

Deep neural network watermarking is a suitable method for protecting the ownership of deep learning (DL) models derived from computationally intensive processes and painstakingly compiled and annotated datasets. It embeds a secret identifier (watermark) within the model, which can be retrieved by the owner to demonstrate ownership. Deepsigns is one of the earliest effective dynamic  white-box watermarking techniques. It has the benefit of maintaining the accuracy of the prediction model without altering the statistical distribution of the model activation maps. In the case of large watermarks, however, DeepSigns watermarked models are vulnerable to multiple attacks. In order to maintain the robustness of DeepSigns, the number of message bits that can be inserted into a model is limited. In this paper, we provide a common framework to formalize the white box watermarking schemes and we propose a novel dynamic white box watermarking scheme "DICTION" that generalizes "Deepsigns". Its originality is derived from adversarial learning using data generated from a latent space. Experiments conducted on the same model test set as Deepsigns demonstrate that our scheme achieves a higher capacity than Deepsigns without sacrificing accuracy or robustness.

Paper preprint is available at http://arxiv.org/abs/2210.15745

# Features
All watermarking schemes and removal attacks are configured for the image classification datasets 
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (32x32 pixels, 10 classes) and [Mnist](https://www.image-net.org) (28x28 pixels, 10 classes) and MNIST Database. 
We implemented the following *watermarking schemes*:
1. [DeepSigns](https://www.microsoft.com/en-us/research/uploads/prod/2018/11/2019ASPLOS_Final_DeepSigns.pdf) 
2. [Uchida](https://dl.acm.org/doi/10.1145/3078971.3078974)
3. [Encryption Resistant scheme](https://ieeexplore.ieee.org/document/9746461)
4. [RIGA](https://dl.acm.org/doi/10.1145/3442381.3450000)
5. [DICTION](https://arxiv.org/abs/2210.15745)

The following attacks are also implemented :

Fine-tuning;
Pruning; 
Overwriting;
Proprety inference attack;

All this defence and attack mecanisms are available to conduct the same expremintaion as us.

# requirements : 
torch~=1.11.0
numpy~=1.21.5
scipy~=1.7.3
matplotlib~=3.4.1
torchvision~=0.12.0
tqdm~=4.60.0

# License :
All codes are provided for research purposes only. When using any code in this project, we would appreciate it if you could refer to this project.

# Contact :
Please send an email to reda.bellafqira@imt-atlantique.fr if you have any questions.

## Cite our paper
```
@article{bellafqira2022diction,
  title={DICTION: DynamIC robusT whIte bOx watermarkiNg scheme},
  author={Bellafqira, Reda and Coatrieux, Gouenou},
  journal={arXiv preprint arXiv:2210.15745},
  year={2022}
}
```
