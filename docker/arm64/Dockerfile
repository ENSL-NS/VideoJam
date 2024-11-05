FROM dustynv/onnxruntime:r35.4.1

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -qq update
RUN apt-get install -qq lsof iperf3 iproute2 tesseract-ocr-eng

# numpy for the newly installed python :
RUN python3 -m pip install \
    numpy \
    aiohttp \
    aiofiles \
    aiocsv \
    tqdm \
    psutil \
    pytesseract

WORKDIR /videojam