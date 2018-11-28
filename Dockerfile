# go from base ubuntu image v 14.04

FROM ubuntu:14.04

COPY snpe-1.21.0/ /snpe-1.21.0/

# lets cover our bases
RUN apt-get update

# install os dependencies
# woah snpe does it for us...

# install neural processing sdk dependencies
RUN ["chmod",  "+x", "snpe-1.21.0/bin/dependencies.sh"]
RUN ["snpe-1.21.0/bin/dependencies.sh"]
RUN ["chmod", "+x", "snpe-1.21.0/bin/check_python_depends.sh"]
RUN ["snpe-1.21.0/bin/check_python_depends.sh"]
