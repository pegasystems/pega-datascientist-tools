# app/Dockerfile

FROM python:3.11.4

WORKDIR /healthcheckapp

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    gdebi-core \
    && rm -rf /var/lib/apt/lists/*
    
# remove -k to allow ssl verification   
RUN curl -k -L https://quarto.org/download/latest/quarto-linux-arm64.deb -o /tmp/quarto-linux-arm64.deb 
RUN gdebi --non-interactive /tmp/quarto-linux-arm64.deb 

# del this line to allow ssl verification on git
RUN git config --global http.sslVerify false  

RUN git clone https://github.com/pegasystems/pega-datascientist-tools.git .

# pip3 install --no-cache-dir .[app], replace with this line to allow ssl verification on pip
RUN pip3 install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org --no-cache-dir .[app]

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "python/pdstools/app/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
