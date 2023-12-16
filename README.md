

# SCALE - Implementation
## SCALE: Constructing Symbolic Comment Trees for Software Vulnerability Detection

## Introduction
Recently, there has been a growing interest in automatic software vulnerability detection. Pre-trained model-based approaches have demonstrated superior performance than other Deep Learning (DL)-based approaches in detecting vulnerabilities. However, the existing pre-trained model-based approaches generally employ code sequences as input during prediction, and may ignore vulnerability-related structural information as reflected in the following two aspects. First, they tend to fail to infer the semantics of the code statements with complex logic such as those containing multiple operators and pointers. Second, they are hard to comprehend various code execution sequences which are important characteristics of vulnerabilities.


To mitigate the challenges, we propose a **S**ymbolic **C**omment tree-based** vulner**A**bi**L**ity d**E**tection framework based on the pre-trained models, named **SCALE**. The proposed Symbolic Comment Tree (SCT) integrates the semantics of code statements with code execution sequences based on the Abstract Syntax Trees (ASTs).



Specifically, SCALE comprises three main modules: (1) *Comment Tree Construction*, which aims at enhancing the model's ability to infer the semantics of code statements by first incorporating Large Language Models (LLMs) for comment generation and then adding the comment node to ASTs. (2) *Symbolic Comment Tree Construction*, which aims at explicitly involving code execution sequence by combining the code syntax templates with the comment tree. (3) *SCT-Enhanced Representation*, which finally incorporates the constructed SCTs for well capturing vulnerability patterns. Experimental results demonstrate that SCALE outperforms the best-performing baseline, including pre-trained model and LLMs, with improvements of 2.96\%, 13.47\%, and 3.75\% in terms of F1 score on the FFMPeg+Qemu, Reveal, and SVulD datasets, respectively. Furthermore, SCALE can be applied to different pre-trained models, such as CodeBERT and UnixCoder, yielding the F1 score performance enhancements ranging from 1.37\% to 10.87\%. 

## Dataset
To investigate the effectiveness of SCALE, we adopt three vulnerability datasets from these paper: 

* FFMPeg+Qemu [1]: [Here](https://drive.google.com/file/d/1LrGV9i5A90qO8S49Bmo3K9AVQyl1sbOI/view?usp=drive_link)
* Reveal [2]: [Here](https://drive.google.com/file/d/1TcV_KzeBWCnAChl92g6vonpNhSVB0H0A/view?usp=drive_link)
* SVulD [3]: [Here](https://drive.google.com/file/d/1fw3SmCJjUCche2cSAhBjjnii7TO3qBje/view?usp=drive_link)
## Requirement
Our code is based on Python3 (>= 3.8). There are a few dependencies to run the code. The major libraries are listed as follows:

* pip install torch (==1.10.2+cu113)
* pip install transformers (==4.34.0)
* pip install numpy (==1.22.3)
* pip install tqdm (==4.63.0)
* pip install sklearn (==1.13)
* pip install tokenizers (==0.14.1)


**Default settings in SCALE**:
* Training configs: 
    * batch_size = 32, lr = 2e-5, epoch = 8

## Comment Gneneration
We use the ChatGPT to generate comment and  the code is under the ```CommentGeneration\```. 

## Symbolic Comment Tree Construction
We use Tree-Sitter to generate the AST and we provide a version of Tree-Sitter [Here](https://drive.google.com/file/d/1JMQbWIgN6GRGRAXW7UdYzD7OVScBK-Fq/view?usp=drive_link). 

The code for symbolic comment tree construction is under the ```SymbolicRule\``` folder. 

## Running the model
The model implementation code is under the ```SCT-Enhanced\``` folder. The model can be runned from ```SCT-Enhanced\train.sh```.

.

## Experiment results
#### Hyper-parameters & Dropout rate / Batch Size


![dropout](https://anonymous.4open.science/r/Comment4Vul2024/Figures/Dropout.png)

<center>Figure1. The impact of dropout rate in different datasets.</center>



![batchsize](https://anonymous.4open.science/r/Comment4Vul2024/Figures/Batchsize.png)

<center>Figure2. The impact of batch size in different datasets.</center>



## References
[1] Yaqin Zhou, Shangqing Liu, Jingkai Siow, Xiaoning Du, and Yang Liu. 2019. Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks. In Advances in Neural Information Processing Systems. 10197–10207.

[2] Saikat Chakraborty, Rahul Krishna, Yangruibo Ding, and Baishakhi Ray. 2020. Deep Learning based Vulnerability Detection: Are We There Yet? arXiv preprint arXiv:2009.07235 (2020).

[3] Chao Ni, Xin Yin, Kaiwen Yang, Dehai Zhao, Zhenchang Xing, and Xin Xia. 2023. Distinguishing Look-Alike Innocent and Vulnerable Code by Subtle Semantic Representation Learning and Explanation. CoRR abs/2308.11237 (2023).