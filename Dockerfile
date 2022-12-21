FROM tensorflow/tensorflow:2.3.0-gpu

ENV WORKDIR /app/

WORKDIR ${WORKDIR}

COPY Pipfile $WORKDIR

RUN pip install --upgrade pip &&  \
	pip install pipenv && \
	pipenv install --system --skip-lock

COPY src/training $WORKDIR/src/training
COPY main.py $WORKDIR/main.py

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