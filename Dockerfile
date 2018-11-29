# go from base ubuntu image v 14.04

FROM ubuntu:18.04

# the greate update
RUN apt-get update
RUN apt-get install -y wget

# 1. install java 
RUN apt-get install -y openjdk-8-jdk

# 2. install android sdk
RUN wget https://dl.google.com/android/repository/sdk-tools-linux-4333796.zip
RUN apt-get install unzip
RUN unzip sdk-tools-linux-4333796.zip
RUN rm sdk-tools-linux-4333796.zip

RUN echo 'hello world'



