#!/bin/sh
# Generate a self-signed certificate on first start, unless one is already
# mounted at /etc/nginx/certs. Runs from nginx's /docker-entrypoint.d before
# the server starts, so a bare `docker compose up` yields a working HTTPS
# listener with no manual openssl work.
#
# Persist /etc/nginx/certs with a volume (compose does) so the certificate
# survives recreates and you only have to trust it once.
#
# AERIE_TLS_HOSTS: comma/space separated names and IPs this box is reached by,
# e.g. "aerie.lan,192.168.1.50". localhost and 127.0.0.1 are always included.
set -eu

CERT_DIR=/etc/nginx/certs
CRT="$CERT_DIR/aerie.crt"
KEY="$CERT_DIR/aerie.key"

if [ -s "$CRT" ] && [ -s "$KEY" ]; then
    echo "aerie: using existing certificate at $CRT"
    exit 0
fi

mkdir -p "$CERT_DIR"

# Always valid for the loopback names, plus whatever the operator declared.
san="DNS:localhost,IP:127.0.0.1,IP:::1"
primary=localhost

for host in $(echo "${AERIE_TLS_HOSTS:-}" | tr ',' ' '); do
    [ -n "$host" ] || continue
    [ "$primary" = localhost ] && primary="$host"
    # Bare IPv4/IPv6 literals have to be IP SANs; browsers ignore DNS SANs for them.
    case "$host" in
        *:*)       san="$san,IP:$host"  ;;  # IPv6 literal
        *[!0-9.]*) san="$san,DNS:$host" ;;  # anything non-numeric is a name
        *)         san="$san,IP:$host"  ;;  # digits and dots: IPv4
    esac
done

echo "aerie: generating self-signed certificate for $san"

# 825 days is the maximum lifetime Safari/iOS accept for a leaf certificate.
openssl req -x509 -newkey rsa:2048 -sha256 -days 825 -nodes \
    -keyout "$KEY" -out "$CRT" \
    -subj "/CN=$primary/O=Aerie/OU=self-signed" \
    -addext "subjectAltName=$san" \
    -addext "basicConstraints=critical,CA:FALSE" \
    -addext "keyUsage=critical,digitalSignature,keyEncipherment" \
    -addext "extendedKeyUsage=serverAuth" \
    >/dev/null 2>&1

chmod 600 "$KEY"
chmod 644 "$CRT"

echo "aerie: certificate written to $CRT (fingerprint below)"
openssl x509 -in "$CRT" -noout -sha256 -fingerprint
