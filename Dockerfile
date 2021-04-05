#PYTHON PART
FROM python:3.7

# Source code
COPY . /app

RUN apt-get update && apt-get install -y unzip

RUN unzip /app/sim_app.zip -d /app

# clean
RUN apt-get autoremove -y && apt-get clean && \
    rm -rf /usr/local/src/*

RUN ["chmod", "+x", "/app/sim_app/linux.x86_64"]

EXPOSE 8090:8080

CMD /app/sim_app/linux.x86_64 & sleep 60
