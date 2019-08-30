############################################################
# Dockerfile to build Python WSGI Application Containers
# Based on Ubuntu
############################################################

# Set the base image to Ubuntu
FROM ubuntu

# File Author / Maintainer
MAINTAINER Shaukat Abidi 

# Update the sources list
RUN apt-get update

# Install Python and Pip
RUN apt-get install -y python3 python3-pip

# Install Flask & Flasgger for web service
RUN pip3 install flask flasgger

# Install required packages
RUN pip3 install requests elasticsearch==6.3.1

ADD /src /src

# Expose ports
EXPOSE 5060

# Set the default directory where CMD will execute
WORKDIR /src

# Set the default command to execute    
# when creating a new container
CMD python3 elasticsearch_service.py
