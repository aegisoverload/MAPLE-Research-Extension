# MAPLE: Metadata-Augmented Private Language Evolution

## Enviroment setup:

First we create a virtual enviroment (here we use conda)
```
conda create -n maple python=3.12
conda activate maple
```
Then follow the step [here](https://docs.vllm.ai/en/v0.10.1.1/getting_started/installation/gpu.html#nvidia-cuda) to install vllm. We use Version: 0.10.1.1.

Then we install PE library. Check [here](https://microsoft.github.io/DPSDA/getting_started/installation.html)
```
pip install "private-evolution[text] @ git+https://github.com/microsoft/DPSDA.git"
```

### TODO
We may possibly missing a few minor package. Double check later.

## Get the dataset
### TODO
We should specify how to get the dataset. For internal use, I've already put all necessary files in the correct directory. The dataset should come with annotated metadata.

### Optional
If you want to annotate the metadata yourself with our schema, please check `./biorxiv/utility_eval/extract_metadata.py` or use `./biorxiv/utility_eval/extract_metadata.sh` to run. By default we use Gemini 2.5 flash-lite. Please change the corresponding part in the code if needed. Remeber to provide API_KEY as well.

## Privatizing metadata
For internal use, we have provided the aim-privatized metadata in the correct folder. So this step can be skipped.

We privatize metadata using AIM. See [here](https://github.com/ryan112358/mbi) for the detail of AIM. 

We recommend to create a new virtual enviroment to install and execute AIM.
```
conda create -n aim python=3.12
conda activate aim
pip install mbi==1.0.0
```

Then you may modify and run `./aim/run_aim.sh` with proper path. Example use:
```
CUDA_VISIBLE_DEVICES=$1 python workplace_biorxiv.py --data_path ../biorxiv/biorxiv_train_metadata_t5emb.csv --save_path [SAVE_PATH] --rho [RHO FOR AIM, CHECK THE PRIVACY_ACCOUNTING PART]
```

For the value of rho, please use `./Privacy_accounting.ipynb` for our pre-computed value. You may also set your own value.

## Run MAPLE

### Precompute the embedding.
```
python ./biorxiv/add_t5emb.py
python ./openreview/add_t5emb.py
```

For dataset = biorxiv/openreview, please check `./{dataset}/main_experiment.sh` if you want to run a specific method and hyperparameter setting. Otherwise you may simply run
```
bash ./{dataset}/main_experiment.sh [YOUR CUDA TO USE.]
```
Note that we recommend to run with at least 2 A100 (40G) GPU. Please adjust the embedding batch size, embedding device (to cpu if only 1 GPU available) accordingly. 

## Evaluation

### For NTP acc evaluation
We use dp-transformer library (Version: 1.0.1) for our implementation. Please create a seperate conda enviroment when running that part of the code.
```
conda create -n dptransformers python=3.12
conda activate dptransformers
```
Then please follow the link [here](https://github.com/microsoft/dp-transformers) for the installation. 



### For biorxiv:

First use `./biorxiv/utility_eval/extract_metadata.sh` to annotate the metadata of the synthetic text. Please check the file for setting up the path correctly.

Then use `./biorxiv/utility_eval/compute_mauve_jsd.sh` and `./biorxiv/utility_eval/fine-tune-nodp-biorxiv_foreval.sh` for computing MAUVE, JSD and training bert-small respectively.

Only executing `./biorxiv/utility_eval/fine-tune-nodp-biorxiv_foreval.sh` will require dp-transformers library, the other evaluations can be done under vllm conda.

### For openreview: 
use `./openreview/downstream_eval_openreview.sh` for training downstream classifier.

### For pmc:
This is the same as the biorxiv case. 

This message is for internal use: since the training file is too large to be uploaded to github, please copy the data from the gcloud machine `~/Mycode/pycode/PMC`. The files that we will use are
```
pmc_train_metadata_aim_eps4.0_(1,9).csv
pmc_train_metadata_aim_eps2.0_(1,9).csv
pmc_train_metadata_aim_eps1.0_(1,9).csv
PMC_V2_train_paired_t5emb.csv
PMC_V2_valid_paired.csv
PMC_V2_test_paired.csv
```
Just copy them the github fold `./MAPLE/pmc` and you may proceed with running the training scripts.

