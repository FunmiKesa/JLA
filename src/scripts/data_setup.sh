# Caltech
cd ../../data
mkdir Caltech
cd Caltech

sh ../src/scripts/gdrive.sh 1yvfjtQV6EnKez6TShMZQq_nkGyY9XA4q set02.tar
sh ../src/scripts/gdrive.sh 1tPeaQr1cVmSABNCJQsd8OekOZIjpJivj set00.tar
sh ../src/scripts/gdrive.sh 1oXCaTPOV0UYuxJJrxVtY9_7byhOLTT8G set08.tar
sh ../src/scripts/gdrive.sh 1jvF71hw4ztorvz0FWurtyCBs0Dy_Fh0A set03.tar
sh ../src/scripts/gdrive.sh 1jvF71hw4ztorvz0FWurtyCBs0Dy_Fh0A set06.tar
sh ../src/scripts/gdrive.sh 1f0mpL2C2aRoF8bVex8sqWaD8O3f9ZgfR set09.tar
sh ../src/scripts/gdrive.sh 1f0mpL2C2aRoF8bVex8sqWaD8O3f9ZgfR set01.tar
sh ../src/scripts/gdrive.sh 1f0mpL2C2aRoF8bVex8sqWaD8O3f9ZgfR set05.tar
sh ../src/scripts/gdrive.sh 1EsAL5Q9FfOQls28qYmr2sO6rha1d4YVz annotations.tar
sh ../src/scripts/gdrive.sh 18TvsJ5TKQYZRlj7AmcIvilVapqAss97X set10.tar
sh ../src/scripts/gdrive.sh 11Q7uZcfjHLdwpLKwDQmr5gT8LoGF82xY set04.tar
sh ../src/scripts/gdrive.sh 1-E_B3iAPQKTvkZ8XyuLcE2Lytog3AofW set07.tar
sh ../src/scripts/gdrive.sh 1h8vxl_6tgi9QVYoer9XcY9YwNB32TE5k cal_labels.zip

cd ..
git clone https://github.com/mitmul/caltech-pedestrian-dataset-converter.git
cd caltech-pedestrian-dataset-converter
python scripts/convert_annotations.py
python scripts/convert_seqs.py
cd ..


mkdir Citypersons
cd Citypersons
sh ../src/scripts/gdrive.sh 1DgLHqEkQUOj63mCrS_0UGFEM9BG8sIZs Citypersons.zip
sh ../src/scripts/gdrive.sh 1BH9Xz59UImIGUdYwUR-cnP1g7Ton_LcZ Citypersons.z01
sh ../src/scripts/gdrive.sh 1q_OltirP68YFvRWgYkBHLEFSUayjkKYE Citypersons.z02
sh ../src/scripts/gdrive.sh 1VSL0SFoQxPXnIdBamOZJzHrHJ1N2gsTW Citypersons.z03

cd ..

mkdir CUHK-SYSU
cd CUHK-SYSU
sh ../src/scripts/gdrive.sh 1D7VL43kIV9uJrdSCYl53j89RE2K-IoQA CUHK-SYSU.zip
cd ..

mkdir PRW
cd PRW
sh ../src/scripts/gdrive.sh 116_mIdjgB-WJXGe8RYJDWxlFnc_4sqS8 PRW.zip
cd ..

mkdir ETHZ
cd ETHZ
sh ../src/scripts/gdrive.sh 19QyGOCqn8K_rc9TXJ8UwLSxCx17e0GoY ETHZ.zip
cd ..

