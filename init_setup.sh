echo "starting the script"

echo "**** Creating and activating env****"
conda create --prefix ./env && source activate ./env
echo "**** Installing dependencies****"
pip install -r requirements.txt
echo "**** Installing Pytorch*****"
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
echo "****exporting the env in conda.yaml file"
conda env export > conda.yaml
echo "*****Instalation completed*******"
