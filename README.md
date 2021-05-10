# Formality Styler
 Web demo for formality classifier (fasttext model) and style transferer (tranformer model trained with fairseq). Code includes training and deployment.

# How to Run
## Prerequisites
### Python Packages
You'll need the following packages to make sure the project runs. You should be able to install the latest version of all the following packages via ```pip install <package name>```.
* [attrs 19.2.0](https://pypi.org/project/attrs/)
* flask
* [fairseq 0.10.2](https://pypi.org/project/fairseq/)
    * make sure ```INPUT_PROMPT``` in [consts.py](consts.py) is correct if using a different version!
* [fold-to-ascii 1.0.2.post1](https://pypi.org/project/fold-to-ascii/)
* [pexpect 4.7.0](https://pypi.org/project/pexpect/)
* [web.py 0.62](https://pypi.org/project/web.py/)
* [fasttext 0.9.2](https://pypi.org/project/fasttext/)
* [requests 2.25.1](https://pypi.org/project/requests/)

### Required Files
The following documents are required:
* Binarized data
* Fairseq models for both directions
* Code file for Byte-pair Encoding (BPE)

You can obtain the zip file containing all these document and put them in the correct structure by running the following commands at project root:
```
$ cd <project root>
$ wget
$ tar
```

### Build Docker Image
First, build the docker image by running
```shell script
cd classifier_docker
docker build -t rewriter .
```
Now the docker image will be ready for classification. 

## Fairseq Model Test Run
To make sure the fairseq model is running as expected, run the following fairseq-cli command at project root. 
```
$ fairseq-interactive data-bin/informal-formal \
                      --path fairseq_results/informal-formal/checkpoint_best.pt \
                      --beam 5 --source-lang informal --target-lang formal \
                      --bpe subword_nmt --bpe-codes data-bin/bpe_code \
                      --tokenizer moses --moses-target-lang en \
                      --remove-bpe
```
The program should print several lines of messages about program parameters, dictionary sizes, and start waiting for user input after this exact string:

```Type the input sentence and press return:```

You should be able to type in an informal sentence, hit enter, and wait for the program to print several lines of rewrite output and probabilities, consisting of (S)ource, (W)all time, .

### Example Interaction
In the example below, run the command, wait for the ```Type the input sentence and press return:``` string to show up, and enter ```hiya, world!```.
```
2021-05-02 23:50:53 | INFO | fairseq.tasks.translation | [informal] dictionary: 10032 types
2021-05-02 23:50:53 | INFO | fairseq.tasks.translation | [formal] dictionary: 9792 types
2021-05-02 23:50:53 | INFO | fairseq_cli.interactive | loading model(s) from fairseq_results/informal-formal/checkpoint_best.pt
2021-05-02 23:50:56 | INFO | fairseq_cli.interactive | NOTE: hypothesis and token scores are output in base 2
2021-05-02 23:50:56 | INFO | fairseq_cli.interactive | Type the input sentence and press return:
hiya, world!
S-0     hiya , world !
W-0     0.242   seconds
H-0     -0.7625648975372314     Hi yes , world !
D-0     -0.7625648975372314     Hi yes, world!
P-0     -2.0801 -1.0306 -0.4909 -0.3452 -0.4730 -0.1554
```
## Starting the System
After the first time set up and tests are done, you can follow this section to start the system. 
### Run Docker Container
To start the Docker container for classifier, run:
```shell script
docker run -p 2500:8081 rewriter
```
If you wish to change the host port number, do the following:
* go to [styler.py](./styler.py) and change the host port ```HOST_PORT``` number
* change the docker run command to the following and run the command
    ```shell script
    docker run -p <HOST_PORT>:8081 rewriter
    ```
Now your docker will be up and listening.
### Open webpage
To start the Flask App, simply run
```shell script
$ python app.py
```
and go to http://127.0.0.1:5000/. If you'd like to specify a port other than 5000, run 
```shell script
$ python app.py --port <port number>
```
go to http://127.0.0.1:\<port number\>/ instead. A Chrome browser is recommended over Safari for a better experience.

# Project Structure



# FAQ
## Why bother with non-ASCII input?

Our model and BPE tokenizer is train on a very clean corpus, in that the input sentences are all ASCII. This means that, even if we assume English-only input from the user, there might be words with diacritics (e.g. Zoë, façade) that will not be recognized by the tokenizer, and the non-ASCII word will be considered as an out-of-vocabulary token and converted into unknown tokens by the tokenizer.

For example, "i think her name's Zoë" will be tokenized into "i think her name ' s Zo\<unk\>", causing the rewriter model to generate unexpected output such as "I think her name is Zoboyfriend." 

Therefore, the restyler addresses this by using a technique similar to [ASCII folding filter in ElasticSearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-asciifolding-tokenfilter.html) using a Python package called [fold-to-ascii](https://pypi.org/project/fold-to-ascii/). To enable ASCII-folding, the user can check the "ASCII-folding" box on the Web UI; the input string from the user will be first and foremost ASCII-folded before being passed to the model (and the classifier if in automatic mode). 

In this way, after ASCII-folding,  "i think her name's Zoë" is first folded into  "i think her name's Zoe", and then fed to the model, which generates the correct out "I think her name is Zoe."

## app.py: Address already in use
When starting [app.py](app.py), messages like ```[Errno 48] Address already in use``` can show up when the port has been occupied due to improper closing. 

To close the port, you'll need to terminate the occupying process: 
* First, find the process occupying it with 
    ```
    $ lsof -i:<port number>
    ```
* In the printed list of processes, locate the process ID (PID) of the Python process occupying this port.
* Use the following command to terminate the process and free up the port:
    ```
    $ kill <PID>
    ```

# Acknowledgements
This project cannot be realized without the generous guidance from Prof. Constantine Lignos. 

The implementation details of this project are also drawn from multiple sources on the internet, so thanks, modern information retrieval technology. 