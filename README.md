

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

## Response-Phase Experiment results

##### Exp 1: Small-scale Experiment of Specific Prompt

| Metrics          | Accuracy | Precision | Recall | F1    |
|:-------------------:|:-------------------:|:-------------------:|:-------------------:| :-------------------:|
| Specific Prompt  | 54.26    | 51.08     | 19.19  | 27.90 |
| SCALE            | <b>66.18    | <b>61.88     | <b>68.69  | <b>65.11 |

<center>Table 1. The small-scale experiment of specific prompt</center>

##### Exp 2: Cost of Calling LLM


|  Dataset   | FFMPeg+Qemu   |  Reveal  | SVulD  |
|:-------------------:|:-------------------:|:-------------------:|:-------------------:| 
| Price    ($)          | 37.73       | 24.14  | 27.21  |
| Response Price   ($)  | 72.61       | 48.64  | 61.39  |
| All Price     ($)     | 110.34      | 72.78  | 88.6   |
| Per Thousand Price ($)| 4.04        | 3.2    | 3.08   |


<center>Table 2. The cost of calling LLM in FFMPeg+Qemu, Reveal and SVulD datasets. </center>

##### Exp 3: Time Cost of Transforming Code

|  Dataset   | FFMPeg+Qemu   |  Reveal  | SVulD  |
|:-------------------:|:-------------------:|:-------------------:|:-------------------:| 
| Per Thousand Time (s)     | 5.91        | 5.37   | 2.41   |


<center>Table 3.The time cost of transforming Code cn FFMPeg+Qemu, Reveal and SVulD datasets.  </center>

##### Exp 4: MCC metric
|  Dataset   | FFMPeg+Qemu   |  Reveal  | SVulD  |
|:-------------------:|:-------------------:|:-------------------:|:-------------------:| 
| CodeBERT   | 0.2348 | 0.4255  | 0.2064  |
| CodeT5     | 0.2749 | 0.4693  | 0.2226  |
| UnixCoder  | 0.2917 | 0.5076  | 0.2571  |
| EPVD       | 0.2586 | 0.496   | 0.2383  |
| LineVul    | 0.2348 | 0.4255  | 0.2501  |
| SvulD      | -      | -       | 0.293   |
| Codellama-7b | 0.0006 | -0.0215 | -0.014  |
| ChatGPT    | 0.0286 | -0.0117 | 0.1361  |
| GPT-3.5-Instruct | 0.0133 | -0.0254 | 0.0776  |
| SCALE      | 0.3263 | 0.5895  | 0.3044  |

<center>Table 4. The MCC evaluation results of SCALE compared with vulnerability detection baselines on the three datasets.</center>

##### Exp 5: T-Test

|  Dataset   | FFMPeg+Qemu   |  Reveal  | SVulD  |
|:-------------------:|:-------------------:|:-------------------:|:-------------------:| 
| CodeT5    | <b>4.22E-02 | <b>4.10E-06 | <b>7.47E-70  |
| UnixCoder | <b>2.67E-08 | 0.54 | <b>4.61E-101 |


<center>Table 5. The p-value of t-test results when SCALE compared with the CodeT5 and UnixCoder in terms of accuracy.</center>


#### Parameter of OPENAI
```
openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301", 
        messages=[
            {"role": "system", "content":prompt},
            {"role": "user", "content": code}], 
        temperature = 0, 
        top_p = 1, 
        frequency_penalty = 0.0, 
        presence_penalty = 0.0,
        stop = ["\n\n"])
```

## References
[1] Yaqin Zhou, Shangqing Liu, Jingkai Siow, Xiaoning Du, and Yang Liu. 2019. Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks. In Advances in Neural Information Processing Systems. 10197–10207.

[2] Saikat Chakraborty, Rahul Krishna, Yangruibo Ding, and Baishakhi Ray. 2020. Deep Learning based Vulnerability Detection: Are We There Yet? arXiv preprint arXiv:2009.07235 (2020).

[3] Chao Ni, Xin Yin, Kaiwen Yang, Dehai Zhao, Zhenchang Xing, and Xin Xia. 2023. Distinguishing Look-Alike Innocent and Vulnerable Code by Subtle Semantic Representation Learning and Explanation. CoRR abs/2308.11237 (2023).