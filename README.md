# ADL-HW2
## Testing
```bash
./download.sh
./run.sh CONTEXT_FILE TEST_FILE OUTPUT_FILE
```
## Reproduce
### 1. Download
Download train, valid, test data to `data` dir.
### 2. Preprocess
```bash
python3.8 preprocess.py
```
### 3. Training
#### Multiple Choice
```bash
cd multiple-choice
./run_no_trainer_base.sh # results will be stored in ../mc_base
./run_no_trainer_large.sh # results will be stored in ../mc_large
cd ..
```
#### Question Answering
```bash
cd question_answering
./run_qa_no_trainer_base.sh # results will be stored in ../qa_base
./run_qa_no_trainer_large.sh # results will be stored in ../qa_large
./run_qa_no_trainer_scratch.sh # results will be stored in ../qa_scratch
cd ..
```
### 4. Inference
Edit parameters to models/tokenizers you would like to test in `run.sh`.
Then run it as what Testing doing.
```bash
./download.sh
./run.sh CONTEXT_FILE TEST_FILE OUTPUT_FILE
```

