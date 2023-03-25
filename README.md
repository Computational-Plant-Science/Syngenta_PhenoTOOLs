# Syngenta_PhenoTOOLs

Author: Suxing Liu,  Alexander Bucksch 


![Maize ear test](../main/media/image_01.png) 

![Maize tassel test](../main/media/image_02.png) 

Robust and parameter-free plant image segmentation and trait extraction.

1. Process with plant image top view, including whole tray plant image.
2. Robust segmentation based on parameter-free color clustering method.
3. Extract individual plant traits, and write output into excel file.

Sample ear test results in Excel format,unit (cm). 

![Sample result of Ear test, unit(cm)](../main/media/image_03.png) 

## Requirements

[Docker](https://www.docker.com/) is required to run this project in a Linux environment.

Install Docker Engine (https://docs.docker.com/engine/install/)

## Usage


1. Build docker image on your PC under linux environment
```shell
docker build -t syngenta_phenotools -f Dockerfile .
```
2. Download prebuild docker image from Docker hub
```shell
docker pull computationalplantscience/syngenta_phenotools
```
3. Run the pipeline inside the docker container 

link your test image path to the /images/ path inside the docker container
 ```shell
docker run -v /path to test image:/images -it syngenta_phenotools
```
(For example: docker run -v /your local directory to cloned "Syngenta_PhenoTOOLs"/Syngenta_PhenoTOOLs/sample_test/Ear_test:/images -it syngenta_phenotools)

4. Run the pipeline inside the container
```shell
python3 trait_computation_mazie_ear.py -p /images/ -ft png

python3 trait_computation_maize_tassel.py -p /images/ -ft png
```
