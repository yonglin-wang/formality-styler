# formality-styler
 Web demo for formality classifier (fasttext model) and style transferer (tranformer model). Code includes training and deployment.

# How to Run
## Prerequisites
### Python Packages
You'll need the following packages to make sure the project runs. You should be able to install the latest version of all the following packages via ```pip install <package name>```.
* [attrs 19.2.0](https://pypi.org/project/attrs/)
* flask
* [fairseq 0.10.2](https://pypi.org/project/fairseq/)
    * make sure ```INPUT_PROMPT``` in [consts.py](consts.py) is correct if using a different version!
* [fold-to-ascii 1.0.2.post1](https://pypi.org/project/fold-to-ascii/)
* [pexpect 4.7.0]()

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

### Open webpage
To start the Flask App, simply run
```
python app.py
```
and go to http://127.0.0.1:5000/. If you'd like to specify a port other than 5000, run 
```
python app.py --port <port number>
```
go to http://127.0.0.1:\<port number\>/ instead. A Chrome browser is recommended over Safari for a better experience.

# Project Structure

# FAQ
## Non-ASCII input?

## app.py: Address already in use
Messages like ```[Errno 48] Address already in use``` are a classic Flask problem where the port has been occupied due to improper closing. 

To close the port, 
* Find the process occupying it with 
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

The implementation details of this project is also drawn from multiple sources on the internet. 
