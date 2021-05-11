# Formality Styler
 Web demo for formality classifier (fasttext model) and style rewriter (tranformer model trained with fairseq). Code includes training and deployment.

# How to Run
## Prerequisites
### Python Packages
You should be able to install the latest version of all the following packages by running the command:
```shell script
pip install -r requirements.txt 
```

The command will install the packages you'll need to run the project, namely:
* [attrs 21.2.0](https://pypi.org/project/attrs/)
* [flask 1.1.2](https://flask.palletsprojects.com/en/1.1.x/)
* [fairseq 0.10.2](https://pypi.org/project/fairseq/)
    * make sure ```INPUT_PROMPT``` in [consts.py](consts.py) is exactly the same as what you see in [Fairseq test run](#fairseq-model-test-run) if using a different version!
* [fold-to-ascii 1.0.2.post1](https://pypi.org/project/fold-to-ascii/)
* [pexpect 4.7.0](https://pypi.org/project/pexpect/)
* [web.py 0.62](https://pypi.org/project/web.py/)
* [requests 2.25.1](https://pypi.org/project/requests/)


### Required Files
The following documents are required:
* Fairseq models for both directions
* Code file and dictionaries for Byte-pair Encoding (BPE)

You can obtain the zip file containing all these document and put them in the correct structure by running the following commands at project root:
```shell script
cd <project root>
wget -O fairseq_data.tar.gz https://www.dropbox.com/s/swllnfve8l6igc7/fairseq_data.tar.gz?dl=1
tar -xf fairseq_data.tar.gz && rm fairseq_data.tar.gz
```
The script above will result in two more directories under project root, [data-bin](data-bin) and [fairseq_results](fairseq_results).

### Build Docker Image
First, [install docker](https://docs.docker.com/get-docker/) and run the Docker daemon. 

Then, build the docker image for the fastText classifier by running
```shell script
cd classifier_docker
docker build -t rewriter .
```
This might take a couple of minutes. Now the docker image will be ready for running. 

## Fairseq Model Test Run
This section is not required for running the project, but is instead intended for testing purpose only.

To test if the fairseq translation model can run as expected, run the following fairseq-cli command at project root. 
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

### Example Interaction with Fairseq
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
After the first time set up and tests are done, you can follow this section to start the system every time you wish to use the system. 
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
python app.py
```
and go to http://127.0.0.1:3500/. If you'd like to specify a port other than 3500, run 
```shell script
python app.py --port <port number>
```
go to http://127.0.0.1:/\<port number\>/ instead. A Chrome browser is recommended over Safari for a better experience.

# Project Structure
## Classifier
This project uses a fastText classifier for predicting the binary formality of the input. 
The classifier is wrapped as an HTTP server with ```web.py``` and ```requests``` and deployed via Docker. 

Files and code involved:
* [classifier_docker](classifier_docker/) for the docker image and server code
* [styler.py](styler.py) for client-side Classifier class

## Rewriter
To rewrite text from informal to formal or from formal to informal, we use ```fairseq-interactive``` to generate output with two transformer models, one for each direction. Interaction with the command line interface is done with ```pexpect```. 

Files and code involved:
* [styler.py](styler.py) for Generator class that spawns child processes and interacts with ```fairseq-interactive```
* [data-bin](data-bin) for decoding the user input text
* [fairseq_results](fairseq_results) for transformer models saved as .pt files

## Web UI
To create a web-based tool that combines the classifier and rewriter on a graphic interface, we use ```flask```.

Files and code involved:
* [templates](templates) for HTML templates
* [app.py](app.py) for handling web UI interaction

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
## Why use Docker for the classifier?
Because python-based fastText refuses to install on my local machine... :( 

In the ideal world, we should be able to dockerize the entire project, but ```fairseq-interactive``` does not recognize ```--path```... we'll look into it when time permits.

# Acknowledgements
This project cannot be realized without the generous guidance from Prof. Constantine Lignos. 

The implementation details of this project are also drawn from multiple sources on the internet, so thanks, modern information retrieval technology. 