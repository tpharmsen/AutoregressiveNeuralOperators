#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 CONTAINER_NAME PROJECTNAME VERSION"
    exit 1
fi

CONTAINER_NAME="$1"
PROJECTNAME="$2"
VERSION="$3"
DOCKERFILE="$4"

# Build the Docker image using the specified container name.
docker build --platform linux/amd64 -t "$CONTAINER_NAME" -f "$DOCKERFILE" .

# Optionally log in if needed:
# docker login https://harbor.spike.tue.nl

# Retrieve the image ID for the built container.
TAG=$(docker images -q "$CONTAINER_NAME")

# Tag the image for the Harbor registry.
docker tag "$TAG" harbor.spike.tue.nl/"$PROJECTNAME"/"$CONTAINER_NAME":"$VERSION"

# Push the image to the Harbor registry.
docker push harbor.spike.tue.nl/"$PROJECTNAME"/"$CONTAINER_NAME":"$VERSION"

 