# TransferLearning-pytorch

# STEPS

## Step 01- Create a repository by using template repository.

## Step 02- Clone the repository

## Step 03- Create a conda environment after opening the repository in VSCODE
```
conda create --prefix ./env python=3.7 -y

```
## Step 04- Activate environment

```conda activate ./env
```
### or
```
source activate ./env
```
## Step 04- install requirements.txt
```
pip install -r requirements.txt
```
## Step 05- install pytorch 11.3

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
## or use init_setup.sh if not want run step 01 to step 05
###in bash terminal use below command
```
bash init_setup.sh
```
### Create Setup.py and install it
```
pip install -e .
```