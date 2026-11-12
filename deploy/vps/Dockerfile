# KCIA WAN pilot: signaling-only image; no PyTorch/GPU/LLM on VPS.
FROM nimlang/nim:2.2.12 AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends libssl-dev ca-certificates && rm -rf /var/lib/apt/lists/*
COPY webrtc_mummy.nimble nim.cfg ./
RUN nimble install -y --depsOnly
COPY src/ ./src/
COPY web/ ./web/
RUN nim c --threads:on --mm:orc -d:release --hints:off -o:/build/kcia-signal src/main.nim

# Use the same Nim base family to avoid runtime libc/SSL ABI mismatches.
FROM nimlang/nim:2.2.12
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl libssl-dev && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 10001 kcia \
 && useradd --system --uid 10001 --gid 10001 --home-dir /var/lib/kcia --shell /usr/sbin/nologin kcia \
 && mkdir -p /var/lib/kcia/rooms && chown -R 10001:10001 /var/lib/kcia
COPY --from=builder /build/kcia-signal /usr/local/bin/kcia-signal
USER 10001:10001
ENV BIND_ADDR=0.0.0.0 PORT=8080 ROOM_DATA_DIR=/var/lib/kcia/rooms
EXPOSE 8080
CMD ["/usr/local/bin/kcia-signal"]
