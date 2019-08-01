#!/bin/sh

CWD=$(cd `dirname $0`; pwd;)

MAPSOCKETS=y
DOCKERFILE=$CWD/docker/bert-google.docker

while test -n "$1" ; do
  case "$1" in
    -h|--help)
      echo "Usage: $0 [--no-map-sockets]" >&2
      exit 1
      ;;
    --no-map-sockets)
      MAPSOCKETS=n
      ;;
    *)
      DOCKERFILE="$1"
      ;;
  esac
  shift
done

# Remap detach to Ctrl+e,e
DOCKER_CFG="/tmp/docker-mrc-nlp-$UID"
mkdir "$DOCKER_CFG" 2>/dev/null || true
cat >$DOCKER_CFG/config.json <<EOF
{ "detachKeys": "ctrl-e,e" }
EOF

set -e -x

docker build \
  --build-arg=http_proxy=$https_proxy \
  --build-arg=https_proxy=$https_proxy \
  --build-arg=ftp_proxy=$https_proxy \
  -t dockerenv \
  -f $DOCKERFILE $CWD/docker


if test "$MAPSOCKETS" = "y"; then
  PORT_TENSORBOARD=`expr 6000 + $UID - 1000`
  PORT_JUPYTER=`expr 8000 + $UID - 1000`
  PORT_OMNIBOARD=`expr 9000 + $UID - 1000`

  # FIXME: Check the port arguments format `ip:porta:portb`. What side does
  # porta/portb belong to?
  DOCKER_PORT_ARGS="-p 0.0.0.0:$PORT_TENSORBOARD:6006 -p 0.0.0.0:$PORT_JUPYTER:8888 -p $PORT_OMNIBOARD:9000 -p 0.0.0.0:7777:7777"
  (
  set +x
  echo
  echo "***************************"
  echo "Host Tensorboard port: ${PORT_TENSORBOARD}"
  echo "Host Jupyter port:     ${PORT_JUPYTER}"
  echo "Host OMNIboard port:   ${PORT_OMNIBOARD}"
  echo "***************************"
  )
fi

# To allow X11 connections from docker
xhost +local: || true
cp "$HOME/.Xauthority" "$CWD/.Xauthority" || true

if which nvidia-docker >/dev/null 2>&1; then
  DOCKER_CMD=nvidia-docker
else
  DOCKER_CMD=docker
fi

${DOCKER_CMD} --config "$DOCKER_CFG" \
    run -it --rm \
    --volume $CWD:/workspace \
    --workdir /workspace \
    -e HOST_PERMS="$(id -u):$(id -g)" \
    -e "CI_BUILD_HOME=/workspace" \
    -e "CI_BUILD_USER=$(id -u -n)" \
    -e "CI_BUILD_UID=$(id -u)" \
    -e "CI_BUILD_GROUP=$(id -g -n)" \
    -e "CI_BUILD_GID=$(id -g)" \
    -e "DISPLAY=$DISPLAY" \
    -e "EDITOR=$EDITOR" \
    -e "TERM=$TERM" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /nix:/nix \
    ${DOCKER_PORT_ARGS} \
    dockerenv \
    bash --login /install/with_the_same_user.sh bash


