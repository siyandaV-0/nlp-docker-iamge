FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set non-interactive mode 
ENV DEBIAN_FRONTEND=noninteractive 

# Copy ubuntu bash script file into working folder
COPY ./dependencies/ubuntu-deps.sh ./

# Install required Ubuntu dependencies
RUN bash ubuntu-deps.sh

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# See http://bugs.python.org/issue19846
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Copy requirements file into working folder
COPY ./dependencies/nlp-requirements.txt ./

#Install requirements for our nlp work env
RUN pip3 install -r nlp-requirements.txt
RUN python3 -m spacy download en_core_web_sm
RUN python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"


# Set Working Directory
WORKDIR /app

# Jupyter-lab localhost runs on port 8888
EXPOSE 8888 

# For better container security use create user
# On the server we are working on UID is 1006:
# ARG USERNAME=container_user
# ARG USER_UID=1006             
# ARG USER_GID=$USER_UID

# RUN groupadd --gid $USER_GID $USERNAME \
#     && useradd --uid $USER_UID --gid $USER_GID -ms /bin/bash $USERNAME 

# USER $USERNAME

# We shall port map the 8888 to port 8181 on a server called bluecrane:
CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter-lab --ip 0.0.0.0 --no-browser --notebook-dir=/app --allow-root --NotebookApp.custom_display_url='http://bluecrane:8181'"]

