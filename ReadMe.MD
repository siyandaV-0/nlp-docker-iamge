<h1 style="Text-Align:center; background-color:DodgerBlue;">
NLP Jupyter Envirnoment Docker Image:
</h1>

---
<h4 style="Text-Align:left;color:black; background-color:Gold;">
    <ul>
    <li>This image builds an NLP environment for use in jupyter-lab <br>
    <li>Deploy the container to a remote server.<br>
    </ul>
</h4>
<br>

<h2 style="Text-Align:left; background-color:MediumSeaGreen;">
Working Directory:
</h2>

<h4 style="Text-Align:left;color:black; background-color:Gold;">
Our working Directory is structured as follows:
</h4>

```
.
└── computer-vision-docker-image
    ├── dependecies
    │   ├── nlp-requirements.txt
    │   └── ubuntu-deps.sh
    ├── Code
    │   └── pretrained_seq_clf.ipynb    
    ├── Dockerfile
    ├── docker-compose.yml
    └── ReadME.md
```
<br>

<h2 style="Text-Align:left; background-color:MediumSeaGreen;">
To build the image run the docker-compose build
</h2>

```shell
 docker-compose build
```
<h4 style="Text-Align:left;color:black; background-color:Gold;">
If you need to make updates to the content of the container, e.g. 
installing new packages with pip- you can specify these in the nlp-requirements.txt file in the dependencies folder.
</h4>

<br> 

 ---

<h2 style="Text-Align:left; background-color:MediumSeaGreen;">
To launch the image run the docker-compose up
</h2>

```shell
 docker-compose up
```
<h4 style="Text-Align:left;color:black; background-color:Gold;">
The docker run bash scripts allow for ease of launching into the container's jupyter-lab. <br> 
This docker run script uses docker-compose commands to achieve this.<br> 
Confirm port in docker-compose file is free to use prior to running the command.<br>
</h4>

<br>


```shell
 docker run --rm --user "container_user" -v "${PWD}/Code:/app" --runtime nvidia -p 8181:8888 -it nlp-env  
```
<h4 style="Text-Align:left;color:black; background-color:Gold;">
In any case ou have issues with using <code style="color:green">docker-compose up</code> which tends to be a case where you cant save your files. <br>
You can opt for this docker run command to launch container's jupyter-lab. <br>  
Confirm port is free to use prior to running the command.<br>
</h4>
