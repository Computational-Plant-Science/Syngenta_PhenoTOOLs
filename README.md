# Syngenta_PhenoTOOLs

Author: Suxing Liu,  Alexander Bucksch 


![Maize ear test](../main/media/image_01.png) 

![Maize tassel test](../main/media/image_02.png) 

Robust and parameter-free plant image segmentation and trait extraction.

1. Process with plant image top view, including whole tray plant image, this tool will segment it into individual images.
2. Robust segmentation based on parameter-free color clustering method.
3. Extract individual plant gemetrical traits, and write output into excel file.

## Requirements

[Docker](https://www.docker.com/) is required to run this project in a Unix environment.

Install Docker Engine (https://docs.docker.com/engine/install/)

## Usage

docker build -t syngenta_phenotools -f Dockerfile .

docker run -v /path to test image:/images -it syngenta_phenotools

cd /opt/Syngenta_PhenoTOOLs/

python3 trait_computation_mazie_ear.py -p /images/ -ft png

python3 trait_computation_maize_tassel.py -p /images/ -ft png
