# Syngenta_PhenoTOOLs

Author: Suxing Liu,  Alexander Bucksch 


![Maize ear test](../main/media/image_01.png) 

![Maize tassel test](../main/media/image_02.png) 

Robust and parameter-free plant image segmentation and trait extraction.

1. Process with plant image top view, including whole tray plant image.
2. Robust segmentation based on parameter-free color clustering method.
3. Extract individual plant traits, and write output into an excel file.

Sample ear test results in Excel format, unit (cm). 

![Sample result of Ear test, unit(cm)](../main/media/image_03.png)

Sample multiple ear test results in Excel format, unit (cm). 

![Sample result of Ear test, unit(cm)](../main/media/image_04.png)



## Imaging requirement:
1. Tassel branches are opened and spread out evenly. 
2. Background was black in color and diffusion reflection material, not reflective.
3. Coin and barcode should be placed under the bottom line of the tassel. 
4. Coin and barcode template images should be cropped and stored in the folder "marker_template" as "barcode.png" and "coin.png" to aid the detection. These template images should be the same for one experiment.
5. Suggested Coin was silver Brazil 1 Real coin, golden ones are not suggested. 
6. Suggested to use a QR code in the future instead of a 2D barcode. 

## Requirements

[Docker](https://www.docker.com/) is required to run this project in a Linux environment.

Install Docker Engine (https://docs.docker.com/engine/install/)

## Usage


1. Build a docker image on your PC under Linux environment
```shell
docker build -t syngenta_phenotools -f Dockerfile .
```
2. Download prebuild docker image from the Docker hub
```shell
docker pull computationalplantscience/syngenta_phenotools
```
3. Run the pipeline inside the docker container 

link your test image path to the /images/ path inside the docker container
 ```shell
docker run -v /path_to_your_test_image:/images -it syngenta_phenotools

or 

docker run -v /path_to_your_test_image:/images -it computationalplantscience/syngenta_phenotools

```
(For example: docker run -v /your local directory to cloned "Syngenta_PhenoTOOLs"/Syngenta_PhenoTOOLs/sample_test/Ear_test:/images -it syngenta_phenotools)

4. Run the pipeline inside the container
```shell
python3 trait_computation_mazie_ear.py -p /images/ -ft png

python3 trait_computation_maize_tassel.py -p /images/ -ft png


Update:

to run mutiple ear test(more than 2 ears, please use "trait_computation_mazie_ear_upgrade.py" and add "-ne 5" paramter)

python3 trait_computation_mazie_ear_upgrade.py -p -p /images/ -ft png -ne 5 -min 250000


```



