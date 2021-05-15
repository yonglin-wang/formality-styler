# Base on Python, for conda use
FROM python:3.7

# Order: commands changed least -> most, e.g. pip -> COPY my_file.py


# installation commands; RUN takes place
# --no-input: can't prompt the user, unintended run
# --no-cache-dir: don't use cache, keep things clean
# Use && to link commands in one RUN command, so that if previous fails, the latter won't run
RUN pip install --no-cache-dir --no-input web.py fasttext requests

COPY classifier.py .
COPY 1gram_ft.ftz .

CMD python classifier.py 8081

