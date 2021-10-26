FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
RUN pip install --upgrade pip
COPY ./requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt
RUN rm ./requirements.txt