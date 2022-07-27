<head>
<!-- Facebook Meta Tags -->
<meta property="og:url" content="https://famousdirector.github.io/DALLEePaperFrame/">
<meta property="og:type" content="website">
<meta property="og:title" content="DALLEePaperFrame">
<meta property="og:description" content="Ever wanted to display never before seen art, on demand, using AI? Press a button, speak a “prompt” for the AI artist, and see the new art!">
<meta property="og:image" content="https://famousdirector.github.io/DALLEePaperFrame/docs/sample.jpg">

<!-- Twitter Meta Tags -->
<meta name="twitter:card" content="summary_large_image">
<meta property="twitter:domain" content="famousdirector.github.io">
<meta property="twitter:url" content="https://famousdirector.github.io/DALLEePaperFrame/">
<meta name="twitter:title" content="DALLEePaperFrame">
<meta name="twitter:description" content="Ever wanted to display never before seen art, on demand, using AI? Press a button, speak a “prompt” for the AI artist, and see the new art!">
<meta name="twitter:image" content="https://famousdirector.github.io/DALLEePaperFrame/docs/sample.jpg">
</head>

# DALLEePaperFrame
Ever wanted to display never before seen art, on demand, using AI?
Press a button, speak a "prompt" for the AI artist, and see the new art!

<img src="docs/sample.jpg" title="Art Generated onto ePaper!">


## What is this project?
By now everyone has seen AI generated art. 
There has been lots of amazing works in this field, perhaps most notably [DALLE2](https://openai.com/dall-e-2/) by OpenAI.
In my opinion, the best way to view art is not on a computer screen, but in a frame on the wall. 

This project use a local server to host the art generation AI and automatic speech recognition capabilities. 
The ePaper frame acts as a client to the server, requesting new art to be generated on demand.

## How does it work?
The server is using an NVIDIA GPU (e.g. a Jetson, or other discrete GPU), and the ePaper frame "client" is running on a Raspberry Pi.
The frame has four buttons, and a microphone.  
The four buttons have the following functions:
1. Request a new generation of art with the same prompt previously used (and currently displayed on the ePaper frame).
2. Request a new generation of art with a new prompt created from the pre-built prompts. (see `prompts.txt`)
3. Request a new generation of art with a new prompt created from the microphone. After the button is pressed, the microphone will start recording for 3 seconds.
4. Enable/disable automatic art generation (based on previously used prompt or pre-built prompts). 

### Extra Technical Details
The display I used was an [Inky Impression 5.7"](https://shop.pimoroni.com/products/inky-impression-5-7?variant=32298701324371) ePaper frame.
It is connected to a [Raspberry Pi 1B+](https://www.raspberrypi.com/products/raspberry-pi-1-model-b-plus/) but any other Raspberry Pi should work.
The `client/` directory contains the single Python script used to control the frame and connect to the server.

The server is running two docker containers, orchestrated by a [Docker Compose](https://docs.docker.com/compose/overview/) file.
The two containers are:
- `triton-inference-server`: Uses NVIDIA's [Triton Inference Server](https://github.com/triton-inference-server) to host the art generation AI model and the automatic speech recognition (ASR) model.
  - The ASR model is a [wav2vec 2.0 Large](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md) model converted to ONNX format for inference.
  - The art generation model is a DALLE-mini variant called [min-dalle](https://github.com/kuprel/min-dalle) (massive shoutout to Brett Kuprel for this incredible Pytorch port).
- `art-generator-api`: a [FastAPI](https://fastapi.tiangolo.com/) server that acts a clean endpoint for the client to request new art.
The `server/` directory contains the code for the server. 

<img src="docs/diagram.jpg" title="Architecture Diagram">

## How do I use this project?
### Set up the Server

Set up the server with the script `setup_server.sh`:
```bash
cd server/
bash setup_server.sh
```

### Run the Server
```bash
cd server/
bash run_server.sh
```

### Set up the Client
Set up the client with the script `setup_client.sh`:
```bash
cd client/
bash setup_client.sh 
```

### Run the Client
```bash
cd client/
bash run_client.sh <ip_address> # the IP address of the server
```

## Final thoughts
### Power considerations
If you want to have this hanging on a wall like I do, you can connect the Raspberry Pi to a cellphone battery pack.

See here for other notes on reducing power consumption on Raspberry Pis:
  - https://www.cnx-software.com/2021/12/09/raspberry-pi-zero-2-w-power-consumption/
  - https://blues.io/blog/tips-tricks-optimizing-raspberry-pi-power/

### System requirements
The min-dalle model and ASR model take around 8GB and 4GB of GPU memory, respectively. So ensure you have at least 12GB of GPU memory. 
If your GPU does not have enough memory, you may want to consider only running the `min-dalle` model for generating art.

### Generation Time
The time taken for art generation is about 10 seconds on an NVIDIA Jetson AGX Orin and about 7 seconds with an NVIDIA RTX2070.

### ePaper
One thing to note about the ePaper is that it is not a perfect display. The particular one I chose has only 7 colors, which can lead to some images looking a bit weird.

Another note is when ePaper refreshes its display. It takes a few seconds to do so. The particular one I chose has a refresh rate of about 30 seconds. Here is a highly sped up sample:

<img src="docs/sample.gif">