FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /workspace

COPY /model/vit_timm /workspace/model/vit_timm

COPY /flask_depo/ /workspace/flask_depo/

COPY imagenet-22k-id2label.json /workspace/

COPY imagenet-22k-label-cn.json /workspace/

WORKDIR /workspace/flask_depo

RUN pip install -r requirements.txt

RUN pip install timm

EXPOSE 5000

# CMD ["python", "app.py"]
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]