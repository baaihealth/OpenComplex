#!/bin/bash

git clone https://github.com/kad-ecoli/rMSA opencomplex/resources/RNA \
  && cd rMSA \
  && ./database/script/update.sh    # Download RNAcentral and nt