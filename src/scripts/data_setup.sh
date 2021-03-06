#!/bin/bash
mkdir models
cd models
sh ../src/scripts/gdrive.sh 1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT ctdet_coco_dla_2x.pth
sh ../src/scripts/gdrive.sh 1iqRQjsG9BawIl8SlFomMg5iwkb6nqSpi fairmot_dla34.pth
sh ../src/scripts/gdrive.sh 1SFOhg_vos_xSYHLMTDGFVZBYjo8cr2fG crowdhuman_dla34.pth
cd ..

cd ./data
mkdir Crowdhuman-files
cd Crowdhuman-files
echo `pwd`

sh ../../src/scripts/gdrive.sh 1tQG3E_RrRI4wIGskorLTmDiWHH2okVvk CrowdHuman_test.zip
sh ../../src/scripts/gdrive.sh 10WIRwu8ju8GRLuCkZ_vT6hnNxs5ptwoL annotation_val.odgt
sh ../../src/scripts/gdrive.sh 1UUTea5mYqvlUObsC1Z8CFldHJAtLtMX3 annotation_train.odgt
sh ../../src/scripts/gdrive.sh 18jFI789CoHTppQ7vmRSFEdnGaSQZ4YzO CrowdHuman_val.zip
sh ../../src/scripts/gdrive.sh 1tdp0UCgxrqy1B6p8LkR-Iy0aIJ8l4fJW CrowdHuman_train03.zip
sh ../../src/scripts/gdrive.sh 17evzPh7gc1JBNvnW1ENXLy5Kr4Q_Nnla CrowdHuman_train02.zip
sh ../../src/scripts/gdrive.sh 134QOvaatwKdy0iIeNqA_p-xkAhkV4F8Y CrowdHuman_train01.zip

unzip CrowdHuman_test.zip
unzip CrowdHuman_val.zip
unzip CrowdHuman_train03.zip
unzip CrowdHuman_train02.zip
unzip CrowdHuman_train01.zip

cd ..

mkdir Caltech-files
cd Caltech-files

echo `pwd`
sh ../../src/scripts/gdrive.sh 1yvfjtQV6EnKez6TShMZQq_nkGyY9XA4q set02.tar
sh ../../src/scripts/gdrive.sh 1tPeaQr1cVmSABNCJQsd8OekOZIjpJivj set00.tar
sh ../../src/scripts/gdrive.sh 1oXCaTPOV0UYuxJJrxVtY9_7byhOLTT8G set08.tar
sh ../../src/scripts/gdrive.sh 1jvF71hw4ztorvz0FWurtyCBs0Dy_Fh0A set03.tar
sh ../../src/scripts/gdrive.sh 1jvF71hw4ztorvz0FWurtyCBs0Dy_Fh0A set06.tar
sjvch ../../src/scripts/gdrive.sh 1f0mpL2C2aRoF8bVex8sqWaD8O3f9ZgfR set09.tar
 
sh ../../src/scripts/gdrive.sh 1f0mpL2C2aRoF8bVex8sqWaD8O3f9ZgfR set05.tar
sh ../../src/scripts/gdrive.sh 1EsAL5Q9FfOQls28qYmr2sO6rha1d4YVz annotations.tar
sh ../../src/scripts/gdrive.sh 18TvsJ5TKQYZRlj7AmcIvilVapqAss97X set10.tar
sh ../../src/scripts/gdrive.sh 11Q7uZcfjHLdwpLKwDQmr5gT8LoGF82xY set04.tar
sh ../../src/scripts/gdrive.sh 1-E_B3iAPQKTvkZ8XyuLcE2Lytog3AofW set07.tar
sh ../../src/scripts/gdrive.sh 1h8vxl_6tgi9QVYoer9XcY9YwNB32TE5k cal_labels.zip

for f in *.tar; do tar xf "$f"; done
unzip cal_labels.zip
cd ..
git clone https://github.com/mitmul/caltech-pedestrian-dataset-converter.git
cd caltech-pedestrian-dataset-converter
ln -s ../Caltech data

python scripts/convert_annotations.py
python scripts/convert_seqs.py
cd ..

wget -O citywalks.zip "https://onedrive.live.com/download?cid=7367E105C63DAC50&resid=7367E105C63DAC50%211637&authkey=AJmkgXYpBLsX-CM"


mkdir Citypersons-files
cd Citypersons-files
sh ../../src/scripts/gdrive.sh 1DgLHqEkQUOj63mCrS_0UGFEM9BG8sIZs Citypersons.zip
sh ../../src/scripts/gdrive.sh 1BH9Xz59UImIGUdYwUR-cnP1g7Ton_LcZ Citypersons.z01
sh ../../src/scripts/gdrive.sh 1q_OltirP68YFvRWgYkBHLEFSUayjkKYE Citypersons.z02
sh ../../src/scripts/gdrive.sh 1VSL0SFoQxPXnIdBamOZJzHrHJ1N2gsTW Citypersons.z03

zip -FF Citypersons.zip --out full.zip 
unzip full.zip

cd ..

mkdir CUHK-SYSU-files
cd CUHK-SYSU-files
sh ../../src/scripts/gdrive.sh 1D7VL43kIV9uJrdSCYl53j89RE2K-IoQA CUHK-SYSU.zip

unzip *.zip
cd ..

mkdir PRW-files
cd PRW-files
sh ../../src/scripts/gdrive.sh 116_mIdjgB-WJXGe8RYJDWxlFnc_4sqS8 PRW.zip
unzip *.zip

cd ..

mkdir ETHZ-files
cd ETHZ-files
sh ../../src/scripts/gdrive.sh 19QyGOCqn8K_rc9TXJ8UwLSxCx17e0GoY ETHZ.zip
unzip *.zip

cd ..

