# Optional provider tools

The production SkyPilot image includes an optional provider-integration
executables:

- `/usr/local/bin/refresh-runpod-catalog.py` refreshes the RunPod GPU catalog.
- `/usr/local/bin/refresh-vast-catalog.py` refreshes the Vast GPU catalog.

The API server also runs an internal `runpod-catalog-refresh-daemon` when
RunPod credentials are configured. It refreshes the shared catalog every 20
minutes by default. The interval can be changed in `~/.sky/config.yaml`:

    daemons:
      runpod-catalog-refresh-daemon:
        interval_seconds: 1200

RunPod marketplace capacity is additionally checked during resource selection,
so a catalog refresh does not need to be perfectly synchronized with a launch.

The utility is not the image default entrypoint. It requires an image built
with the RunPod extra and a persistent `/root` volume for its catalog output.

Run the catalog utility directly with a persistent /root volume:

    docker run --rm \
      --env-file ./skypilot-server.env \
      --volume skypilot-server-root:/root \
      IMAGE \
      /usr/local/bin/refresh-runpod-catalog.py

The API server runs an internal `vast-catalog-refresh-daemon` when the Vast
credential file is configured. It refreshes the catalog every 20 minutes by
default. Set
`daemons.vast-catalog-refresh-daemon.interval_seconds` in
`~/.sky/config.yaml` to change the interval.

The utility is not the image default entrypoint. It requires an image built
with the Vast extra and a persistent `/root` volume for its catalog output.

Run the catalog utility directly with a persistent `/root` volume:

```bash
docker run --rm \
  --env-file ./skypilot-server.env \
  --volume skypilot-server-root:/root \
  IMAGE \
  /usr/local/bin/refresh-vast-catalog.py
```
