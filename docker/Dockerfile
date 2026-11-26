FROM python:3.13-slim-trixie@sha256:98ce6fb13b3803744fbca11434f00818f1e300e4e48a4f204e07b2c928b6218e
COPY requirements.lock /opt/requirements.lock
RUN pip install --no-cache-dir --require-hashes --only-binary=:all: -r /opt/requirements.lock && pip check
COPY runner.py /opt/runner.py
USER 65534:65534
ENV PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /output
ENTRYPOINT ["python", "-I", "/opt/runner.py"]
