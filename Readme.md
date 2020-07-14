# Retrieval-based Neural Source Code Summarization
This project mainly includes source ode, experimental data and results of our paper entitled "Retrieval-based Neural Source Code Summarization" that has been published at ICSE'2020. Instead of only relying on a single encoder-decoder model, our proposed approach Rencos can take advantages of both neural and retrieval-based techniques by enhancing it with retrieved similar code snippets from the aspects of syntax and semantics though our fusion model. The results on the source code summarization task show that it can effectively deal with low-frequency problem in NMT-based approach and thus improve the performance.



## How to run the code and reproduce experimtal results of our approach?
### Requirements
* Hardwares: 16 cores of 2.4GHz CPU, 128GB RAM and a Titan Xp GPU with 12GB memory (GPU is a must )
* OS: Ubuntu 16.04
* Packages:
	+ python 3.6 (for runing the main code)
	+ pytorch 0.4.1
	+ torchtext 0.3.1
	+ nltk 3.2.4
	+ ConfigArgParse 0.14.0
	+ pylucene 7.4.0 (see http://lucene.apache.org/pylucene/install.html)
	+ python 2.7 (only for evaluation, we recommand a conda environment for switching the python version)

### Quick start
If all the requirements are met, just run:
`python run.py all python 2` or `python run.py all java 2` for python and java datasets respectively.
The results will be saved in file "samples/(python|java)/output/test.out"

### Step through
1. Preprocess
	For python dataset (PCSD):
	`python run.py preprocess python`
	For java dataset (JCSD):
	`python run.py preprocess java`

2. Train
	`python run.py train python`
	Or
	`python run.py train java`
	
3. Retrieval
	`python run.py retrieval python`
	Or
	`python run.py retrieval java`
	
	The results will be saved in files "samples/(python|java)/output/ast.out" and "samples/(python|java)/output/rnn.out", corresponding to Only syntactic level retrieval and Only semantic level retrieval respectively.
4. Generation
	* To obtain the results of Rencos, run the following command
		`python run.py translate python 2`
		Or
		`python run.py translate java 2`
	
	* To obtain the results of NMT-only, run the following command
		`python run.py translate python 0`
		Or
		`python run.py translate java 0`
	
	The results will be saved in file "samples/(python|java)/output/test.out". We also provide the results of NMT and Rencos corresponding to baseline.out and rencos.out respectively.
### Automatic Evaluation

(*Switch into python 2.7*)

First,
	`cd evaluation/`
	
Evaluate the results on PCSD:
`python evaluate.py ../samples/python/output/test.out ../samples/python/test/test.txt.tgt 50`

Evaluate the results on JCSD:
`python evaluate.py ../samples/java/output/test.out ../samples/java/test/test.txt.tgt 30`
Here 50 and 30 are the length limit.

	




