FROM dl.nvidia/cuda:10.1.cudnn7

# RUN pip3 install --no-cache-dir Pillow==8.1.2 && \
#     pip3 install -U memory_profiler && \
#     pip3 install --no-cache-dir visdom==0.1.8.8 &&\
#     pip3 install --no-cache-dir dominate==2.4.0 &&\
#     pip3 install --no-cache-dir torchstat &&\
#     rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache

COPY ./detectron2 /facebookresearch/detectron2

RUN cd /facebookresearch && python3 -m pip install -e detectron2

WORKDIR /facebookresearch/detectron2/projects/PointRend/

ENTRYPOINT ["python3", "train_net.py"]