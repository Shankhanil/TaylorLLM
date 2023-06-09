# LLM to generate Taylor Swift style songs 
##### Assignment of Outsystems 

The code to generate a Taylor Swift style songs, finetuned on the GPT2 transformer LLM.

## Dataset
The dataset used can be found in this [link](https://www.kaggle.com/datasets/ishikajohari/taylor-swift-all-lyrics-30-albums)

## Setting up environment

Setup the conda environment using the ``environment.yaml`` file using the following command given below.

```bash
conda env create -f environment.yml
conda activate taylorllm
```

## Generating lyrics

Clone the repository

```bash
git clone git@github.com:Shankhanil/TaylorLLM.git
cd TaylorLLM
```
Download the checkpoints file from this [link](https://drive.google.com/drive/folders/1L4mF9gztWnkw30C_mgd5GEe1kUYSjoCO?usp=sharing). Place the ``.pt`` file in the ``checkpoints/`` folder. 

Edit the ``config.py`` file. Give the specifications of the source text file and the output text file in the ``SOURCE_PATH`` and ``DUMP_PATH`` respectively. Give the specifications of ``CHECKPOINT_PATH``. Then run the following python command

```bash
python test.py
```

The output will be saved in the location specified in ``DUMP_PATH``.

## Running the finetuning script
To run the finetuning script as it was, just run the ``run.sh`` file using the following command

```bash
sh run.sh
```

However, if you want to specify model parameters, please look for the description of the arguments using 
```bash
python main.py -h
```


## Feedbacks
Feedbacks are hugely welcome. Please register bugs, issues, feedbacks and suggestions in the Issues section. Feedbacks are also welcome through [email](shankha.rik@gmail.com).

## License
See ``LICENSE.md`` for License details