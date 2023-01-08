# multitouch_attribution

This is a playground to test and implement different marketing attribution models based on ML or algorithmic approaches.
The underlying dataset is hosted at [Kaggle](https://www.kaggle.com/kavitabhagwani/marketing-campaign).
<br><br>
Different attribution model approaches are wrapped in some scripted notebooks at `notebooks`, where real
jupyter notebooks are exported from and moved to `exported_notebooks`. Referring functions and source
code is stored at the projects base folder called `multitouch_attribution`. <br>
A first overview can be achieved with a common *EDA* step (`01_customer_journey_eda.py`).

## Contact

Danny Stumpe <danny.stumpe@gmail.com>

## Dev Setup

1. Make sure `Docker` and `docker-compose` are installed. It is highly recommended to run the code in a Docker container.

    ### Other options
    If you do not want to follow these steps using Docker, you can stop here and explore the project on your own using local environments such as `poetry` or `venv`.

    Creating a local environment using `poetry` can be reached via

    ```bash
    poetry install
    ```
    ( assuming that `poetry` is already installed and `pyproject.toml` is located at the current working directory)

    To use a local environment, just run
    ``` bash
    python -m venv <my_venv_name>
    source ./<my_venv_name>/bin/activate
    ```

2. Next, create the Docker image using `docker-compose`
    ```bash
    docker-compose build ide
    ```

3. Make sure your SSH agent is running
    ```bash
    eval "$(ssh-agent -s)"
    ```

    NOTE:
    The image build will fail if you don't use SSH, since the `docker-compose.yaml` is designed to mount `$SSH_AUTH_SOCK:/tmp/authsock` in order to make git
    available within the container.
    If you don't want to use this feature, you can remove `SSH_AUTH_SOCK` mountning from `docker-compose.yaml`. This would also entail modification in the `Dockerfile` according to your needs.

4. After successfully building the image, create the container based on that image
    ```bash
    docker-compose up ide
    ```

5. Now, the browser-like IDE `code-server` should be running and can be reached via `localhost:8123`.

6. After accessing the browser IDE, the correct interpreter is probably not yet selected. Consequently, this has to be done manually (`Ctrl` + `Shift`+ `P` $\rightarrow$ type `"Python: Select Interpreter"` $\rightarrow$ navigate to the poetry python interpreter at `/root/.cache/pypoetry/virtualenvs/<project_venv>/bin/python`).

7. Finally, create a new terminal and remove the recently active one.
Now, everything is prepared to run the code using the correct environment.
