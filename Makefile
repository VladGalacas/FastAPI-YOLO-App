IMAGE_NAME := fastapi-project
CONTAINER_NAME := fastapi-project-container
HOST ?= 0.0.0.0
PORT ?= 8000
DEVICE ?= cuda

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run --gpus all -d \
        -p $(PORT):$(PORT) \
        -e DEVICE=$(DEVICE) \
        -e APP_HOST=$(HOST) \
        -e APP_PORT=$(PORT) \
        --name $(CONTAINER_NAME) \
        $(IMAGE_NAME)

stop-rm:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

rm-image:
	docker rmi $(IMAGE_NAME) || true

logs:
	docker logs -f $(CONTAINER_NAME)