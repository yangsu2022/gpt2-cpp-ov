Here we provide a C++ demo about GPT2. Optimize the post-processing through the custom node of the OpenVINO ppp API, that is, insert some custom operations like TopK. Thereby reducing the latency of the pipeline.

GPT2 Introduction
------
We use the  [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf ) model, which is a part of the Generative Pre-trained Transformer (GPT) family. GPT-2 is pre-trained on a large corpus of English text using unsupervised training. 
The following image illustrates complete demo pipeline used for this scenario:
![image](https://github.com/yangsu2022/gpt2-cpp-ov/assets/102195992/ea85461b-a378-411b-851a-581c6430f20b)

Implementation
------
GPT2 tokenizer:
The C++ implementation of the GPT2 tokenizer is from the repo [gpt2-cpp](https://github.com/gf712/gpt2-cpp).  

GPT2 OpenVINO model:
The ONNX model is downloaded during the build process from the [ONNX Model Zoo](https://github.com/onnx/models/tree/main/text/machine_comprehension/gpt-2) repo.

Notice:
  * Download ONNX model with Git LFS
  * Use the [python script](https://github.com/onnx/models/blob/main/text/machine_comprehension/gpt-2/dependencies/GPT2-export.py) to export the ONNX Model with ORT 1.10. 
  * The model is 634 MB large, so it may take a while to download it.
  * Use python openvino-dev to convert ONNX model to OpenVINO IR model via "mo -m gpt2-lm-head-10.onnx -o IR_FP32"

## Installation
The cmake script will look for ONNX header files and dynamic library using the repo structure based from the $HOME directory. After compiling make sure that the dynamic library can be found by the runtime, ie. set `LD_LIBRARY_PATH` accordingly. So replace it with OpenVINO.

The other three dependencies are already included in this project:
  * [simdjson](https://github.com/simdjson/simdjson)
  * [ctre](https://github.com/hanickadot/compile-time-regular-expressions)
  * [cxxopts](https://github.com/jarro2783/cxxopts/)
  * [cpptqdm](https://github.com/aminnj/cpptqdm/blob/master/tqdm.h)

Compiling the binary requires a C++17 compliant compiler.
Additionally the ONNX model is downloaded during the build process from the [ONNX Model Zoo](https://github.com/onnx/models) repo. The model is 634 MB large, so it may take a while to download it :)

The vocabulary and merges files are provided in this repository, but were originally obtained from the [transformers](https://github.com/huggingface/transformers) repo.

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

```bash
cd build
./gpt2-generate -h
Usage:
  GPT2 [OPTION...]

  -t, --text arg    Initial text for GPT2
  -n, --number arg  Number of new words to generate from initial text
                    (default: 1)
  -p, --ppp         Optimize with PPP for topK
  -h, --help        Print usage


./gpt2-generate -t "If I have 100 million U.S. dollars, how can I maximize my returns in today's artificial intelligence world?" -n 128 -p
Optimizing model with PrePostProcessor
using OpenVINO runtime with topK sampling(k=20)

  ████████████████████████████████████████▏ 100.0% [ 128/ 128 | 12.4 Hz | 10s<0s]  
MEAN time of single token prediction: 75.0312 ms
OV Prediction: "If I have 100 million U.S. dollars, how can I maximize my returns in today's artificial intelligence world? I don't know!"


./gpt2-generate -t "If I have 100 million U.S. dollars, how can I maximize my returns in today's artificial intelligence world?" -n 128 
using OpenVINO runtime with topK sampling(k=20)

  ████████████████████████████████████████▏ 100.0% [ 128/ 128 | 7.6 Hz | 17s<0s]  
MEAN time of single token prediction: 126.289 ms
OV Prediction: "If I have 100 million U.S. dollars, how can I maximize my returns in today's artificial intelligence world? Well, what about the rest of the world? How could you optimize your returns with the same technology?

It turns out this approach isn't just limited to the big two markets. The other way to increase your return is by making it simpler. By starting with a single-purpose solution, such as a simple-to-use app, which you can integrate into any existing application, you get an additional layer of efficiency. For instance, instead of having to add new features every time you use it, there's the possibility that you'll start seeing better results every time you're using it. As a result, this approach makes"


The world has become more complex and more complex over the past few decades. There have been many changes to the nature of technology and what makes it successful. In many countries there are large-scale artificial intelligence systems, including a few in Germany, which may be able to do better than the human brain can. But as more and more people move from the use of artificial intelligence and computer-based artificial intelligence systems (AI) to more sophisticated AI and machine learning, they are also increasingly more dependent on human skills to perform their jobs. While we can all benefit from our own understanding of human intelligence and machine"

```
