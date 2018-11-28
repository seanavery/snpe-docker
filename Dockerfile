# go from base ubuntu image v 14.04

FROM ubuntu:14.04

COPY snpe-1.21.0/ /snpe-1.21.0/

# install neural processing sdk dependencies
RUN ["chmod",  "+x", "snpe-1.21.0/bin/dependencies.sh"]
RUN ["snpe-1.21.0/bin/dependencies.sh"]
