# R 4.2 environment for the relative-success replication.
#
# The upstream pipeline is run unchanged as an environment check: if Figure 1
# reproduces, the R shim, the target-indication ingest and the relative-success
# baseline are all validated in one step.
#
# The tidyverse base image is used rather than r-ver because building tidyverse
# from source costs an hour and produces the same result.

FROM rocker/tidyverse:4.2.3

RUN apt-get update && apt-get install -y --no-install-recommends \
        libcurl4-openssl-dev \
        libssl-dev \
        libxml2-dev \
        libglpk-dev \
    && rm -rf /var/lib/apt/lists/*

RUN install2.r --error --skipinstalled --ncpus -1 \
        binom \
        DescTools \
        epitools \
        glue \
        janitor \
        lawstat \
        openxlsx \
        optparse \
        readxl \
        reshape2 \
        weights \
        writexl

WORKDIR /work
