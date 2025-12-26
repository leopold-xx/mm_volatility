mkdir -p "/Volumes/T7 Shield/data/binance_usd_xrpusdt" && \
rsync -avhP --partial --inplace \
  -e "ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=120 -o TCPKeepAlive=yes -o Compression=yes" \
  bin-230:/home/sixian/binance_usd_xrpusdt/ \
  "/Volumes/T7 Shield/data/binance_usd_xrpusdt/"