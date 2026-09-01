FROM pgvector/pgvector:pg17

ARG PG_TEXTSEARCH_VERSION=1.4.0
ARG TARGETARCH

RUN set -eux; \
    case "${TARGETARCH}" in \
        amd64) package_arch="amd64"; package_sha256="93dbb144b09675ce5294d2a8655ed6b7f53a79cb7ebee1b7c8c3c148561a0383" ;; \
        arm64) package_arch="arm64"; package_sha256="c084c942caa9d6e35a84aaff8b21e6c51afa4126030aabf1d5e76f03f4f2a320" ;; \
        *) echo "Unsupported Docker architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl unzip; \
    package="pg-textsearch-v${PG_TEXTSEARCH_VERSION}-pg17-${package_arch}.zip"; \
    curl -fsSL \
        "https://github.com/timescale/pg_textsearch/releases/download/v${PG_TEXTSEARCH_VERSION}/${package}" \
        -o "/tmp/${package}"; \
    echo "${package_sha256}  /tmp/${package}" | sha256sum -c -; \
    unzip "/tmp/${package}" -d /tmp/pg_textsearch; \
    apt-get install -y --no-install-recommends /tmp/pg_textsearch/*.deb; \
    rm -rf /tmp/pg_textsearch "/tmp/${package}"; \
    apt-get purge -y --auto-remove curl unzip; \
    rm -rf /var/lib/apt/lists/*
