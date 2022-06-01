#FROM continuumio/anaconda3 as base
#
#WORKDIR /app
#
## Make RUN commands use `bash --login`:
#SHELL ["/bin/bash", "--login", "-c"]
#
## Create the environment:
#RUN conda create -n myenv python=3.9
#
#RUN conda init bash
#
#RUN echo "conda activate myenv" >> ~/.bashrc
#SHELL ["/bin/bash", "--login", "-c"]
#
#RUN conda install -c conda-forge tensorflow-gpu
#RUN conda install -c conda-forge matplotlib
#RUN conda install -c conda-forge scikit-learn
#
#COPY ./src /app/src
#
#FROM base as development
## The code to run when container is started:
#ENTRYPOINT ["python", "/app/src/main.py"]
##Z

# Define base image
FROM continuumio/miniconda3

# Set working directory for the project
WORKDIR /app

# Create Conda environment from the YAML file
COPY environment.yml .
RUN conda env create -f environment.yml

# Override default shell and use bash
SHELL ["conda", "run", "-n", "Keras", "/bin/bash", "-c"]

# Python program to run in the container
COPY . .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "Keras", "python", "/app/src/main.py"]
