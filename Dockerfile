FROM tensorflow/tensorflow:2.3.0-gpu

WORKDIR /app/

COPY Pipfile Pipfile

RUN pip install --upgrade pip &&  \
	pip install pipenv && \
	pipenv install --system --skip-lock

COPY src/training src/training
COPY main.py main.py

COPY data /root/.ros/uneven-ground-driving-result

# 利用するdockerイメージのPythonバージョンが3.6なので、
# Python 3.9以降用の型アノテーションを削除する必要がある
RUN find . -name "*.py" | xargs sed -i.bak -e "s/ \-> tuple\[.*\]//g" && \
	find . -name "*.py" | xargs sed -i.bak -e "s/: tuple\[.*\],/,/g" && \
    find . -name "*.py" | xargs sed -i.bak -e "s/: tuple\[.*\])/)/g" && \
    find . -name "*.py" | xargs sed -i.bak -e "s/: tuple\[.*\] //g" && \
    find . -name "*.py" | xargs sed -i.bak -e "s/ \-> list\[.*\]//g" && \
	find . -name "*.py" | xargs sed -i.bak -e "s/: list\[.*\],/,/g" && \
    find . -name "*.py" | xargs sed -i.bak -e "s/: list\[.*\])/)/g" && \
    find . -name "*.py" | xargs sed -i.bak -e "s/: list\[.*\] //g"