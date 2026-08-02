# MAPLE: Metadata-Augmented Private Language Evolution (COLM 2026)

## Environment setup:

First, we create a virtual environment (here we use conda)
```
conda create -n maple python=3.12
conda activate maple
```
Then follow the step [here](https://docs.vllm.ai/en/v0.10.1.1/getting_started/installation/gpu.html#nvidia-cuda) to install vllm. We use Version: 0.10.1.1.
```
pip install vllm==0.10.1.1 --extra-index-url https://download.pytorch.org/whl/cu126
```
Then we install PE library. Check [here](https://microsoft.github.io/DPSDA/getting_started/installation.html)
```
pip install "private-evolution[text] @ git+https://github.com/microsoft/DPSDA.git"
```

Finally, we install some minor packages

```
pip install datasets==4.0.0
```

## Get the dataset
We already put all necessary files in the correct directory. The dataset should come with annotated metadata.

### Optional
If you want to annotate the metadata yourself with our schema, please check `./biorxiv/utility_eval/extract_metadata.py` or use `./biorxiv/utility_eval/extract_metadata.sh` to run. By default, we use Gemini 2.5 flash-lite. Please change the corresponding part in the code if needed. Remember to provide API_KEY as well.

## Privatizing metadata
We have provided the aim-privatized metadata in the correct folder. So this step can be skipped. However, you may also run AIM yourself, following the steps below.

We privatize metadata using AIM. See [here](https://github.com/ryan112358/mbi) for the detail of AIM. 

We recommend creating a new virtual environment to install and execute AIM.
```
conda create -n aim python=3.12
conda activate aim
pip install mbi==1.0.0
```

Then you may modify and run `./aim/run_aim.sh` with the proper path. Example use:
```
CUDA_VISIBLE_DEVICES=$1 python workplace_biorxiv.py --data_path ../biorxiv/biorxiv_train_metadata_t5emb.csv --save_path [SAVE_PATH] --rho [RHO FOR AIM, CHECK THE PRIVACY_ACCOUNTING PART]
```

## Run MAPLE

### Precompute the embedding.
```
python ./biorxiv/add_t5emb.py
python ./openreview/add_t5emb.py
```

For dataset = biorxiv/openreview, please check `./{dataset}/main_experiment.sh` if you want to run a specific method and hyperparameter setting. Otherwise, you may simply run
```
bash ./{dataset}/main_experiment.sh [YOUR CUDA TO USE.]
```
Note that we recommend running with at least 2 A100 (40G) GPUs. Please adjust the embedding batch size and embedding device (to cpu if only 1 GPU available) accordingly. 

## Evaluation

### For NTP acc evaluation
We use dp-transformer library (Version: 1.0.1) for our implementation. Please create a seperate conda enviroment when running that part of the code.
```
conda create -n dptransformers python=3.12
conda activate dptransformers
```
Then please follow the link [here](https://github.com/microsoft/dp-transformers) for the installation. 



### For biorxiv:

First, use `./biorxiv/utility_eval/extract_metadata.sh` to annotate the metadata of the synthetic text. Please check the file for setting up the path correctly.

Then use `./biorxiv/utility_eval/compute_mauve_jsd.sh` and `./biorxiv/utility_eval/fine-tune-nodp-biorxiv_foreval.sh` for computing MAUVE, JSD, and training bert-small respectively.

Only executing `./biorxiv/utility_eval/fine-tune-nodp-biorxiv_foreval.sh` will require the dp-transformers library; the other evaluations can be done under vLLM conda.

### For openreview: 
Use `./openreview/downstream_eval_openreview.sh` for training the downstream classifier.


## Contact

If you have any questions, please open an issue and/or reach out to Eli Chien (elichientwn@gmail.com) and Yuzheng Hu (yh46@illinois.edu).

## Citation
If you find our code or work useful, please consider citing our paper (will be updated to COLM citation after the camera-ready version is made public):
```
@article{chien2026maple,
  title={Maple: Metadata augmented private language evolution},
  author={Chien, Eli and Hu, Yuzheng and McKenna, Ryan and Wu, Shanshan and Xu, Zheng and Kairouz, Peter},
  journal={arXiv preprint arXiv:2603.19258},
  year={2026}
}
```
