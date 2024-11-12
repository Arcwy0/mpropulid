# MagazinePro alpha

## Docker server hint

There are two Dockerfiles in the repo. The file for working with the server is `Dockerfile_server`.

Build:
```
docker build -f Dockerfile_server -t pulid-server .
```
Run server:
```
docker run -it --name pulid_server --gpus '"device=0"' -p 8011:8080 -v /media/imit-learn/ISR_2T/MPro/PuLID:/app pulid-server \
    python3 server_main_fastapi.py --offload --fp8
```

Usual commands to start/stop:
```
docker stop pulid_server
docker rm pulid_server
```

Previously it ran on ngrok (ngrok needs to be installed):
```
ngrok http http://localhost:8011
```

All files with name starting with 'server' are server files:

1. `server_main_fastapi.py` — main file in which server pipeline is configured.
2. `server_email_sender.py` — file in which email sending is configured.
3. `server_process_images.py` — file in which model is loaded and images are processed.
4. `server_queue_manager.py` — works with `queue/pending.json` file and loads the user requests to the model.