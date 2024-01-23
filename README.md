# Enhancing Generalizability in Protein-Ligand Binding Affinity Prediction with Multimodal Contrastive Learning

## Requirements

To set up your environment to run the code, install the following packages:

matplotlib==3.7.2  
networkx==3.1  
numpy==1.24.3  
pandas==2.0.3  
pymol==0.1.0  
rdkit==2022.09.5  
scikit_learn==1.3.0  
scipy==1.5.2  
seaborn==0.12.2  
pytorch==1.12.1  
torch_geometric==2.0.3  
tqdm==4.63.0  
openbabel==3.1.1  
pymol-open-source==2.5.0  


## Usage

### 1. Prepare a Dataset

We provide a toy dataset to demonstrate the training and testing of our model.

**Dataset structure:**
```
data/
toy_set/
    ligand/
    ligand_1.sdf
    ligand_2.sdf
    ...
    protein/
    protein_1.pdb
    protein_2.pdb
    ...
```
CSV file format:
```
pdb,affinity
3uri,9
4m0z,5.19
4kz6,3.1
4jxs,4.74
2r9w,5.1
...
```
**Preprocessing steps:**

1. Run the preprocessing script: `python preprocessing.py`

2. Prepare the dataset: `python dataset_ConBAP.py`


### 2. Model Training

**Contrastive Learning with Redocked 2020 Dataset:**

- Process the redocked 2020 dataset. (The processed data sets are available at [here](https://doi.org/10.5281/zenodo.10532672).)
- Run the pretraining script: `python pretrain.py`.


**Fine-Tuning with PDBbind Dataset:**

- A checkpoint for contrastive learning is available in `./unsupervised/model`.
- Run the training script: `python train_ConBAP.py` (The processed data sets are available at [here](https://doi.org/10.5281/zenodo.10532672).)
(Note: Modify file paths based on your directory structure.)

### 3. Model Testing

- Testing checkpoints are located in `./supervised/model`.
- Run the prediction script: `python predict_single.py`.
- If you want to test the docking power or screening power in CASF-2016:
- Run the test script: `python casf_docking_single.py` `casf_screening_single.py`.

(Note: Modify file paths based on your directory structure.)
## Reference

- The paper has been submitted for publication.

