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

# install java 8
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;

RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

