#!/bin/sh

docker run --name "$CONTAINER_NAME" --rm -u $(id -u):$(id -g) -v "$( cd "$( dirname "$0" )" && pwd )":/code -v "$( cd "$( dirname "$0" )" && pwd )":/results task3
