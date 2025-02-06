# Create a Python 3.8 environment:
# > conda create -n trade python==3.8

conda install -c conda-forge -c defaults \
    absl-py \
    appdirs \
    astroid \
    astunparse \
    backcall \
    black \
    cachetools \
    certifi \
    charset-normalizer \
    click \
    cloudpickle \
    cycler \
    dataclasses \
    debugpy \
    decorator \
    Deprecated \
    dm-tree \
    gast \
    google-auth \
    google-auth-oauthlib \
    google-pasta \
    grpcio \
    h5py \
    idna \
    inflection \
    ipykernel \
    ipython \
    isort \
    jedi \
    joblib \
    Keras-Preprocessing \
    kiwisolver \
    kt-legacy \
    lazy-object-proxy \
    lxml \
    Markdown \
    matplotlib \
    matplotlib-inline \
    mccabe \
    more-itertools \
    multipledispatch \
    nest-asyncio \
    numpy \
    oauthlib \
    packaging \
    pandas \
    pandas-datareader \
    parso \
    pathspec \
    pexpect \
    pickleshare \
    Pillow \
    pip \
    platformdirs \
    prompt-toolkit \
    protobuf \
    psutil \
    ptyprocess \
    pyasn1 \
    pyasn1-modules \
    Pygments \
    pylint \
    pyparsing \
    python-dateutil \
    pytz \
    pyzmq \
    Quandl \
    regex \
    requests \
    requests-oauthlib \
    rsa \
    scikit-learn \
    scipy \
    setuptools \
    six \
    tabulate \
    tensorboard \
    tensorboard-data-server \
    tensorboard-plugin-wit \
    tensorflow \
    tensorflow-estimator \
    tensorflow-probability \
    termcolor \
    threadpoolctl \
    toml \
    tomli \
    tornado \
    traitlets \
    typing_extensions \
    urllib3 \
    wcwidth \
    Werkzeug \
    wheel \
    wrapt

conda install anaconda::mypy_extensions

conda install conda-forge::gpflow

conda install conda-forge::empyrical

conda install conda-forge::yfinance